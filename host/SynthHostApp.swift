// SynthRL host window. Plays the synth in real time:
//   on-screen keys -> MIDI -> SynthAudioUnit -> speakers
//   output tap     -> oscilloscope + peak readout
import AppKit
import AudioToolbox
import AVFAudio
import Combine
import SwiftUI
import UniformTypeIdentifiers

private let synthDescription = AudioComponentDescription(
    componentType: synthComponentType,
    componentSubType: synthComponentSubtype,
    componentManufacturer: synthComponentManufacturer,
    componentFlags: 0,
    componentFlagsMask: 0)

/// How an imported sample becomes a timbre. Analysis is the step 2a pitch and
/// spectrum method; Model runs the trained step 2b network from a checkpoint.
enum SourceMode: String, CaseIterable, Identifiable {
    case analysis = "Analysis"
    case model = "Model"
    var id: String { rawValue }
}

@MainActor
final class AudioHost: ObservableObject {
    let model = SynthModel()
    @Published var sourceMode: SourceMode = .analysis
    @Published private(set) var isImporting = false
    private let engine = AVAudioEngine()
    private var unit: AVAudioUnit?
    private var pendingTimbre: (name: String, values: [Float], envelope: [Float])?

    private static let projectRootKey = "projectRoot"
    private static let checkpointKey = "checkpointPath"

    func start() async {
        guard unit == nil else { return }
        AUAudioUnit.registerSubclass(SynthAudioUnit.self,
                                     as: synthDescription,
                                     name: "SynthRL",
                                     version: 1)
        do {
            // Load in-process. Without this, a copy of the .appex inside this
            // app is launched as a separate service, which fails (code 4097).
            let unit = try await AVAudioUnit.instantiate(with: synthDescription,
                                                         options: [.loadInProcess])
            let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 2)!
            engine.attach(unit)
            engine.connect(unit, to: engine.mainMixerNode, format: format)

            let scope = model.scope
            engine.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { buffer, _ in
                guard let data = buffer.floatChannelData else { return }
                let n = Int(buffer.frameLength)
                scope.update(Array(UnsafeBufferPointer(start: data[0], count: n)))
            }

            // Push the UI's values into the unit before audio starts.
            if let tree = unit.auAudioUnit.parameterTree {
                for (index, value) in model.params.enumerated() {
                    tree.parameter(withAddress: AUParameterAddress(index))?.value = value
                }
            }

            try engine.start()
            self.unit = unit
            model.sendParam = { [weak self] index, value in
                self?.unit?.auAudioUnit.parameterTree?
                    .parameter(withAddress: AUParameterAddress(index))?.value = value
            }
            model.sendMIDI = { [weak self] bytes in
                guard let block = self?.unit?.auAudioUnit.scheduleMIDIEventBlock else { return }
                var b = bytes
                b.withUnsafeMutableBufferPointer { p in
                    _ = block(AUEventSampleTimeImmediate, 0, p.count, p.baseAddress!)
                }
            }
            let rate = engine.outputNode.outputFormat(forBus: 0).sampleRate
            model.status = "ONLINE · \(Int(rate)) Hz"
            NSLog("SynthRL audio running at %f Hz", rate)
            if let pending = pendingTimbre {
                pendingTimbre = nil
                apply(timbre: pending.values, envelope: pending.envelope, name: pending.name)
            }
        } catch {
            let e = error as NSError
            NSLog("SynthRL audio start failed: domain=%@ code=%d %@", e.domain, e.code, "\(e)")
            model.status = "ERROR · \(e.localizedDescription) (\(e.code))"
        }
    }

    /// Opens a timbre file made by `python -m neural.export` and loads it into the
    /// synth. The host is unsandboxed, so it reads the file itself; the AU only
    /// receives the parsed harmonics.
    func loadTimbre() {
        guard let url = pickFile(title: "Choose a timbre file made by neural.export",
                                 types: [.json]) else { return }
        loadTimbreFile(url, name: url.lastPathComponent)
    }

    /// Picks a WAV sample, analyses it in the project's Python environment (the
    /// same `neural.export` command, run as a local process) and loads the timbre
    /// it writes. The shipped plugin never runs Python; this is the prototype path.
    func importSample() {
        guard !isImporting else { return }
        guard let picked = pickFile(title: "Choose a sample to analyse (WAV, AIFF, MP3, M4A, FLAC…)",
                                    types: [.audio]) else { return }
        guard let root = projectRoot() else { return }
        var checkpoint: URL?
        if sourceMode == .model {
            guard let saved = savedCheckpoint() ?? chooseCheckpoint() else { return }
            checkpoint = saved
        }

        isImporting = true
        model.status = "ANALYSING · \(picked.lastPathComponent)"
        Task {
            // Temporary files this import creates; removed when it finishes.
            var temporary: [URL] = []
            defer {
                temporary.forEach { try? FileManager.default.removeItem(at: $0) }
                isImporting = false
            }
            do {
                let wav = try await wavForAnalysis(picked)
                if wav != picked { temporary.append(wav) }

                let out = FileManager.default.temporaryDirectory
                    .appendingPathComponent("synthrl-\(UUID().uuidString).json")
                temporary.append(out)
                var args = [wav.path, out.path]
                if let checkpoint {
                    args += ["--checkpoint", checkpoint.path]
                }
                let summary = try await runProcess(
                    root.appendingPathComponent(".venv/bin/python"),
                    args: ["-m", "neural.export"] + args, cwd: root)
                NSLog("SynthRL import: %@", summary)
                loadTimbreFile(out, name: picked.lastPathComponent)
            } catch {
                let reason = error.localizedDescription
                if sourceMode == .analysis && reason.contains("no clear pitch") {
                    model.status = "IMPORT ERROR · no clear pitch (try Source: Model for unpitched sounds)"
                } else {
                    model.status = "IMPORT ERROR · \(reason)"
                }
            }
        }
    }

    /// The WAV that analysis reads. WAV input is used as it is. Any other audio
    /// format is converted with afconvert, which ships with macOS, to a temporary WAV.
    private func wavForAnalysis(_ input: URL) async throws -> URL {
        if input.pathExtension.lowercased() == "wav" { return input }
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("synthrl-\(UUID().uuidString).wav")
        do {
            _ = try await runProcess(URL(fileURLWithPath: "/usr/bin/afconvert"),
                                     args: ["-f", "WAVE", "-d", "LEI16", input.path, output.path],
                                     cwd: nil)
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw NSError(domain: "SynthRL", code: 2, userInfo: [NSLocalizedDescriptionKey:
                "could not convert \(input.lastPathComponent) to WAV (\(error.localizedDescription))"])
        }
        return output
    }

    /// Asks for a trained model checkpoint and remembers it for Model imports.
    @discardableResult
    func chooseCheckpoint() -> URL? {
        guard let url = pickFile(title: "Choose a trained model checkpoint (.pt)",
                                 types: [UTType(filenameExtension: "pt") ?? .data]) else { return nil }
        UserDefaults.standard.set(url.path, forKey: Self.checkpointKey)
        model.status = "CHECKPOINT · \(url.lastPathComponent)"
        return url
    }

    /// Parses a timbre file and applies it, or holds it until audio is running.
    private func loadTimbreFile(_ url: URL, name: String) {
        do {
            let data = try Data(contentsOf: url)
            guard let doc = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  doc["format"] as? String == "synth-timbre",
                  let values = doc["harmonics"] as? [Double] else {
                throw NSError(domain: "SynthRL", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "not a synth-timbre file"])
            }
            // Version 2 files carry a loudness envelope; version 1 files do not.
            let envelope = (doc["envelope"] as? [Double] ?? []).map { Float($0) }
            let harmonics = values.map { Float($0) }
            // Audio may still be starting (the unit is created asynchronously).
            // Keep the timbre and apply it as soon as audio is running.
            guard unit != nil else {
                pendingTimbre = (name, harmonics, envelope)
                model.status = "TIMBRE · waiting for audio: \(name)"
                return
            }
            apply(timbre: harmonics, envelope: envelope, name: name)
        } catch {
            model.status = "TIMBRE ERROR · \(error.localizedDescription)"
        }
    }

    private func pickFile(title: String, types: [UTType]) -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.canChooseDirectories = false
        panel.message = title
        return panel.runModal() == .OK ? panel.url : nil
    }

    /// The synthRL folder that holds `.venv`. Asked for once, then remembered.
    private func projectRoot() -> URL? {
        func usable(_ url: URL) -> Bool {
            FileManager.default.isExecutableFile(
                atPath: url.appendingPathComponent(".venv/bin/python").path)
        }
        if let saved = UserDefaults.standard.string(forKey: Self.projectRootKey) {
            let url = URL(fileURLWithPath: saved)
            if usable(url) { return url }
        }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.message = "Choose the synthRL folder (the one containing .venv)"
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        guard usable(url) else {
            model.status = "IMPORT ERROR · no .venv/bin/python in that folder"
            return nil
        }
        UserDefaults.standard.set(url.path, forKey: Self.projectRootKey)
        return url
    }

    private func savedCheckpoint() -> URL? {
        guard let saved = UserDefaults.standard.string(forKey: Self.checkpointKey),
              FileManager.default.fileExists(atPath: saved) else { return nil }
        return URL(fileURLWithPath: saved)
    }

    /// Runs a command-line tool and returns its standard output. Runs off the main
    /// thread, so the UI stays responsive during analysis or conversion.
    private func runProcess(_ executable: URL, args: [String], cwd: URL?) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = executable
                process.arguments = args
                process.currentDirectoryURL = cwd
                let stdout = Pipe()
                let stderr = Pipe()
                process.standardOutput = stdout
                process.standardError = stderr
                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: error)
                    return
                }
                // Drain both pipes before waiting, so a full pipe cannot stall the tool.
                let outData = stdout.fileHandleForReading.readDataToEndOfFile()
                let errData = stderr.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                if process.terminationStatus == 0 {
                    let text = String(decoding: outData, as: UTF8.self)
                    continuation.resume(returning: text.trimmingCharacters(in: .whitespacesAndNewlines))
                } else {
                    // The last stderr line is the tool's own error; earlier lines are library warnings.
                    let text = String(decoding: errData, as: UTF8.self)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let message = text.split(separator: "\n").last.map(String.init) ?? "analysis failed"
                    continuation.resume(throwing: NSError(
                        domain: "SynthRL", code: Int(process.terminationStatus),
                        userInfo: [NSLocalizedDescriptionKey: message]))
                }
            }
        }
    }

    /// Pushes a timbre into the running unit and makes it audible.
    private func apply(timbre: [Float], envelope: [Float], name: String) {
        guard let au = unit?.auAudioUnit else {
            model.status = "TIMBRE ERROR · audio is not running"
            return
        }
        // Go through AUAudioUnit's full state rather than casting to SynthAudioUnit:
        // the unit's class comes from the bundled extension, not from this app.
        // An envelope key left over from an earlier timbre must not survive a
        // version 1 file, so it is removed when this file has no envelope.
        var state = au.fullState ?? [:]
        state[synthTimbreStateKey] = timbre.map { Double($0) }
        if envelope.isEmpty {
            state.removeValue(forKey: synthTimbreEnvelopeStateKey)
        } else {
            state[synthTimbreEnvelopeStateKey] = envelope.map { Double($0) }
        }
        au.fullState = state
        let source = Int(SYNTH_PARAM_SOURCE_MIX)
        au.parameterTree?.parameter(withAddress: AUParameterAddress(source))?.value = 1
        model.params[source] = 1
        model.status = "TIMBRE · \(name)"
    }
}

@main
struct SynthHostApp: App {
    @StateObject private var host = AudioHost()

    var body: some Scene {
        WindowGroup("SynthRL") {
            ContentView(model: host.model)
                .task { await host.start() }
                .toolbar {
                    Picker("Source", selection: $host.sourceMode) {
                        ForEach(SourceMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(host.isImporting)
                    Button("Import Sample…") { host.importSample() }
                        .disabled(host.isImporting)
                    Button("Checkpoint…") { host.chooseCheckpoint() }
                    Button("Load Timbre…") { host.loadTimbre() }
                        .disabled(host.isImporting)
                }
        }
        // Open at the content's ideal size so everything fits without scrolling.
        .windowResizability(.contentMinSize)
    }
}
