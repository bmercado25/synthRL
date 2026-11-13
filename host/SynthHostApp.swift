// SynthRL host window. Plays the synth in real time:
//   on-screen keys -> MIDI -> SynthAudioUnit -> speakers
//   output tap     -> oscilloscope + peak readout
import AppKit
import AudioToolbox
import AVFAudio
import Combine
import SwiftUI

private let synthDescription = AudioComponentDescription(
    componentType: synthComponentType,
    componentSubType: synthComponentSubtype,
    componentManufacturer: synthComponentManufacturer,
    componentFlags: 0,
    componentFlagsMask: 0)

/// Latest output samples, written by the audio tap and read by the UI.
final class ScopeBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var samples = [Float](repeating: 0, count: 1024)
    private var peak: Float = 0

    func update(_ new: [Float]) {
        let p = new.reduce(Float(0)) { max($0, abs($1)) }
        lock.lock()
        samples = new
        peak = p
        lock.unlock()
    }

    func snapshot() -> (samples: [Float], peak: Float) {
        lock.lock()
        defer { lock.unlock() }
        return (samples, peak)
    }
}

@MainActor
final class AudioHost: ObservableObject {
    @Published var status = "Starting audio…"
    @Published var activeNote: Int?
    @Published var velocity: Double = 100

    let scope = ScopeBuffer()
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

            let scope = self.scope
            engine.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { buffer, _ in
                guard let data = buffer.floatChannelData else { return }
                let n = Int(buffer.frameLength)
                scope.update(Array(UnsafeBufferPointer(start: data[0], count: n)))
            }

            try engine.start()
            self.unit = unit
            let rate = engine.outputNode.outputFormat(forBus: 0).sampleRate
            status = "Running · \(Int(rate)) Hz · stereo"
            NSLog("SynthRL audio running at %f Hz", rate)
        } catch {
            let e = error as NSError
            NSLog("SynthRL audio start failed: domain=%@ code=%d %@", e.domain, e.code, "\(e)")
            status = "Audio error: \(e.localizedDescription) (code \(e.code))"
        }
    }

    func noteOn(_ note: Int) {
        guard let block = unit?.auAudioUnit.scheduleMIDIEventBlock else { return }
        send(block, [0x90, UInt8(note), UInt8(velocity)])
        activeNote = note
    }

    /// Only releases the note if it is still the one we started, so a fast
    /// slide from one key to the next does not cut the new note.
    func noteOff(_ note: Int) {
        guard let block = unit?.auAudioUnit.scheduleMIDIEventBlock,
              activeNote == note else { return }
        send(block, [0x80, UInt8(note), 0])
        activeNote = nil
    }

    private func send(_ block: AUScheduleMIDIEventBlock, _ bytes: [UInt8]) {
        var b = bytes
        b.withUnsafeMutableBufferPointer { p in
            _ = block(AUEventSampleTimeImmediate, 0, p.count, p.baseAddress!)
        }
    }
}

func noteName(_ midi: Int) -> String {
    let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return "\(names[midi % 12])\(midi / 12 - 1)"
}

struct ScopeView: View {
    let buffer: ScopeBuffer

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30)) { _ in
            let snap = buffer.snapshot()
            Canvas { ctx, size in
                var axis = Path()
                axis.move(to: CGPoint(x: 0, y: size.height / 2))
                axis.addLine(to: CGPoint(x: size.width, y: size.height / 2))
                ctx.stroke(axis, with: .color(.gray.opacity(0.4)), lineWidth: 1)

                var wave = Path()
                let s = snap.samples
                for (i, v) in s.enumerated() {
                    let x = size.width * CGFloat(i) / CGFloat(s.count - 1)
                    let y = size.height / 2 - CGFloat(v) * size.height / 2 * 0.9
                    if i == 0 { wave.move(to: CGPoint(x: x, y: y)) } else { wave.addLine(to: CGPoint(x: x, y: y)) }
                }
                ctx.stroke(wave, with: .color(.green), lineWidth: 1.5)
            }
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(alignment: .topTrailing) {
                Text(String(format: "peak %.2f", snap.peak))
                    .font(.caption.monospaced())
                    .foregroundStyle(.green)
                    .padding(8)
            }
        }
    }
}

struct KeyboardView: View {
    @ObservedObject var host: AudioHost

    // One octave, C4 to C5.
    private let whites = [60, 62, 64, 65, 67, 69, 71, 72]
    // Each black key sits on the boundary after white key index `after`.
    private let blacks: [(note: Int, after: Int)] = [(61, 0), (63, 1), (66, 3), (68, 4), (70, 5)]

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width / CGFloat(whites.count)
            let bw = w * 0.6
            let bh = geo.size.height * 0.6
            ZStack(alignment: .topLeading) {
                HStack(spacing: 0) {
                    ForEach(whites, id: \.self) { note in
                        key(note, isBlack: false)
                            .frame(width: w, height: geo.size.height)
                    }
                }
                ForEach(blacks, id: \.note) { b in
                    key(b.note, isBlack: true)
                        .frame(width: bw, height: bh)
                        .offset(x: CGFloat(b.after + 1) * w - bw / 2)
                }
            }
        }
    }

    private func key(_ note: Int, isBlack: Bool) -> some View {
        let pressed = host.activeNote == note
        let fill: Color = pressed ? .accentColor : (isBlack ? .black : .white)
        return RoundedRectangle(cornerRadius: 4)
            .fill(fill)
            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.gray))
            .overlay(alignment: .bottom) {
                Text(noteName(note))
                    .font(.caption2)
                    .foregroundStyle(isBlack && !pressed ? Color.white : Color.black)
                    .padding(4)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if host.activeNote != note { host.noteOn(note) }
                    }
                    .onEnded { _ in host.noteOff(note) }
            )
    }
}

struct ContentView: View {
    @ObservedObject var host: AudioHost

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("SynthRL · Rudimentary Sine")
                    .font(.title2.bold())
                Text(host.status)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            ScopeView(buffer: host.scope)
                .frame(height: 170)

            KeyboardView(host: host)
                .frame(height: 200)

            HStack(spacing: 12) {
                Text("Velocity")
                Slider(value: $host.velocity, in: 1...127)
                Text("\(Int(host.velocity))")
                    .monospacedDigit()
                    .frame(width: 34, alignment: .trailing)
                Spacer()
                Text(host.activeNote.map { "Playing \(noteName($0))" } ?? "Idle")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(20)
        .frame(minWidth: 680, minHeight: 540)
    }
}

@main
struct SynthHostApp: App {
    @StateObject private var host = AudioHost()

    var body: some Scene {
        WindowGroup("SynthRL") {
            ContentView(host: host)
                .task { await host.start() }
        }
    }
}
