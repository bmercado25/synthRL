// AUv3 integration test. Acts like a host:
//   1. registers the AUAudioUnit subclass in-process
//   2. instantiates it through AVAudioUnit (the normal host path)
//   3. sends MIDI through the unit's scheduleMIDIEventBlock
//   4. renders offline and checks what comes out
import AudioToolbox
import AVFoundation
import Foundation

var failures = 0

func check(_ ok: Bool, _ message: String) {
    print(ok ? "  ok   \(message)" : "  FAIL \(message)")
    if !ok { failures += 1 }
}

let desc = AudioComponentDescription(
    componentType: synthComponentType,
    componentSubType: synthComponentSubtype,
    componentManufacturer: synthComponentManufacturer,
    componentFlags: 0,
    componentFlagsMask: 0)

AUAudioUnit.registerSubclass(SynthAudioUnit.self,
                             as: desc,
                             name: "SynthRL Test",
                             version: 1)

let sampleRate = 44_100.0
let blockFrames: AVAudioFrameCount = 512

/// Creates engine + AU, returns the pieces a host would hold.
func makeHost() throws -> (AVAudioEngine, AVAudioUnit) {
    let semaphore = DispatchSemaphore(value: 0)
    var instrument: AVAudioUnit?
    var instantiateError: Error?
    AVAudioUnit.instantiate(with: desc, options: [.loadInProcess]) { unit, error in
        instrument = unit
        instantiateError = error
        semaphore.signal()
    }
    semaphore.wait()
    if let instantiateError { throw instantiateError }
    guard let unit = instrument else {
        throw NSError(domain: "test", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "no AU instance"])
    }

    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
    let engine = AVAudioEngine()
    engine.attach(unit)
    engine.connect(unit, to: engine.mainMixerNode, format: format)
    try engine.enableManualRenderingMode(.offline, format: format,
                                         maximumFrameCount: 4096)
    try engine.start()
    return (engine, unit)
}

/// Sends a MIDI message right now (sample time "immediate").
func sendMIDI(_ unit: AVAudioUnit, _ status: UInt8, _ d1: UInt8, _ d2: UInt8) -> Bool {
    guard let schedule = unit.auAudioUnit.scheduleMIDIEventBlock else { return false }
    var bytes: [UInt8] = [status, d1, d2]
    bytes.withUnsafeMutableBufferPointer { ptr in
        _ = schedule(AUEventSampleTimeImmediate, 0, 3, ptr.baseAddress!)
    }
    return true
}

/// Renders `seconds` of audio. Returns interleaved-by-channel arrays.
func render(_ engine: AVAudioEngine, seconds: Double) throws -> [[Float]] {
    let format = engine.manualRenderingFormat
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: blockFrames)!
    let total = Int(seconds * sampleRate)
    var left: [Float] = []
    var right: [Float] = []
    left.reserveCapacity(total)
    right.reserveCapacity(total)
    while left.count < total {
        let status = try engine.renderOffline(blockFrames, to: buffer)
        guard status == .success else {
            throw NSError(domain: "test", code: Int(status.rawValue),
                          userInfo: [NSLocalizedDescriptionKey: "render status \(status)"])
        }
        let n = Int(buffer.frameLength)
        left.append(contentsOf: UnsafeBufferPointer(start: buffer.floatChannelData![0], count: n))
        right.append(contentsOf: UnsafeBufferPointer(start: buffer.floatChannelData![1], count: n))
    }
    return [left, right]
}

func peak(_ x: [Float]) -> Float { x.map { abs($0) }.max() ?? 0 }

func frequency(_ x: [Float]) -> Double {
    var crossings = 0
    for i in 1..<x.count where x[i - 1] <= 0 && x[i] > 0 { crossings += 1 }
    return Double(crossings) * sampleRate / Double(x.count)
}

func writeWAV(_ channels: [[Float]], to path: String) throws {
    // WAV files are interleaved, so build an interleaved buffer.
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                               channels: 2, interleaved: true)!
    let frames = AVAudioFrameCount(channels[0].count)
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
    buffer.frameLength = frames
    let dst = buffer.floatChannelData![0]
    for i in 0..<channels[0].count {
        dst[2 * i] = channels[0][i]
        dst[2 * i + 1] = channels[1][i]
    }
    let file = try AVAudioFile(forWriting: URL(fileURLWithPath: path),
                               settings: format.settings,
                               commonFormat: .pcmFormatFloat32,
                               interleaved: true)
    try file.write(from: buffer)
}

do {
    print("Host instantiates the AU")
    let (engine, unit) = try makeHost()
    check(unit.auAudioUnit.componentDescription.componentSubType == synthComponentSubtype,
          "instantiated the expected component")
    check(unit.auAudioUnit.scheduleMIDIEventBlock != nil,
          "exposes a MIDI input block")

    print("Silence before any note")
    let silent = try render(engine, seconds: 0.1)
    check(peak(silent[0]) == 0, "no output without input")

    print("MIDI note on (A4) produces a 440 Hz tone")
    check(sendMIDI(unit, 0x90, 69, 100), "note-on sent")
    let tone = try render(engine, seconds: 1.0)
    check(peak(tone[0]) > 0.1, "output is audible")
    check(abs(frequency(tone[0]) - 440) < 2, "frequency is 440 Hz (got \(String(format: "%.1f", frequency(tone[0]))))")
    check(tone[0] == tone[1], "left and right channels match")

    print("Note off returns to silence")
    check(sendMIDI(unit, 0x80, 69, 0), "note-off sent")
    _ = try render(engine, seconds: 0.05)  // let the release fade finish
    let released = try render(engine, seconds: 0.1)
    check(peak(released[0]) == 0, "silent after release")

    print("Deterministic: a fresh host renders identical audio")
    let (engine2, unit2) = try makeHost()
    _ = sendMIDI(unit2, 0x90, 69, 100)
    let again = try render(engine2, seconds: 1.0)
    check(again == tone, "bit-identical to the first run")

    let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "build/au_output.wav"
    try writeWAV(tone, to: out)
    print("  wrote \(out)")
} catch {
    print("  FAIL unexpected error: \(error)")
    failures += 1
}

print("\n\(failures == 0 ? "PASSED" : "FAILED") (\(failures) failure\(failures == 1 ? "" : "s"))")
exit(failures == 0 ? 0 : 1)
