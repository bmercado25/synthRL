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

@MainActor
final class AudioHost: ObservableObject {
    let model = SynthModel()
    private let engine = AVAudioEngine()
    private var unit: AVAudioUnit?

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
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.canChooseDirectories = false
        panel.message = "Choose a timbre file made by neural.export"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let data = try Data(contentsOf: url)
            guard let doc = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  doc["format"] as? String == "synth-timbre",
                  let values = doc["harmonics"] as? [Double] else {
                throw NSError(domain: "SynthRL", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "not a synth-timbre file"])
            }
            guard let synth = unit?.auAudioUnit as? SynthAudioUnit else {
                throw NSError(domain: "SynthRL", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "audio is not running yet"])
            }
            synth.loadTimbre(values.map { Float($0) })
            model.params[Int(SYNTH_PARAM_SOURCE_MIX)] = 1
            model.status = "TIMBRE · \(url.lastPathComponent)"
        } catch {
            model.status = "TIMBRE ERROR · \(error.localizedDescription)"
        }
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
                    Button("Load Timbre…") { host.loadTimbre() }
                }
        }
        // Open at the content's ideal size so everything fits without scrolling.
        .windowResizability(.contentMinSize)
    }
}
