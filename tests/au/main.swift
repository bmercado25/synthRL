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
    // Pump the main run loop while waiting: the completion is delivered on the
    // main queue, so blocking the main thread on the semaphore would deadlock.
    let deadline = Date().addingTimeInterval(30)
    while semaphore.wait(timeout: .now()) != .success {
        if Date() > deadline { throw NSError(domain: "test", code: 2, userInfo: [NSLocalizedDescriptionKey: "instantiate timed out"]) }
        RunLoop.main.run(until: Date().addingTimeInterval(0.01))
    }
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

    print("Parameter tree exposes the schema")
    let tree = unit.auAudioUnit.parameterTree!
    let expected: [(AUParameterAddress, String)] = [
        (0, "oscA_wave"), (1, "oscA_level"), (2, "oscB_wave"), (3, "oscB_level"),
        (4, "filter_cutoff"), (5, "filter_resonance"),
        (6, "env_attack"), (7, "env_decay"), (8, "env_sustain"), (9, "env_release"),
        (10, "oscA_detune"), (11, "oscB_detune"),
        (12, "filter_env_amount"), (13, "filter_env_attack"), (14, "filter_env_decay"),
        (15, "filter_env_sustain"), (16, "filter_env_release"),
        (17, "lfo_rate"), (18, "lfo_pitch_depth"), (19, "lfo_filter_depth"),
        (20, "drive"), (21, "reverb_mix"), (22, "reverb_size"),
    ]
    for (address, identifier) in expected {
        check(tree.parameter(withAddress: address)?.identifier == identifier,
              "address \(address) is \(identifier)")
    }
    let waveParam = tree.parameter(withAddress: 0)!
    check(waveParam.valueStrings?.count == 4, "waveform offers 4 choices")
    check(waveParam.valueStrings?.last == "Triangle", "last waveform is Triangle")

    print("Parameter changes reach the audio")
    // Each scenario uses a fresh host so earlier settings do not leak in.
    func rendered(_ setup: (AUParameterTree) -> Void) throws -> [Float] {
        let (engine, host) = try makeHost()
        setup(host.auAudioUnit.parameterTree!)
        _ = sendMIDI(host, 0x90, 69, 127)
        return try render(engine, seconds: 1.0)[0]
    }
    func rms(_ x: [Float]) -> Double {
        sqrt(x.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(x.count))
    }

    let defaultOut = try rendered { _ in }
    check(abs(rms(defaultOut) - 0.5 / 2.0.squareRoot()) < 0.01, "default is a sine (RMS 0.354)")

    let squareOut = try rendered { $0.parameter(withAddress: 0)?.value = 2 }
    check(abs(rms(squareOut) - 0.5) < 0.01, "osc A set to Square (RMS 0.5)")

    let silentOut = try rendered { $0.parameter(withAddress: 1)?.value = 0 }
    check(peak(silentOut) == 0, "osc A level 0 is silent")

    let bOut = try rendered {
        $0.parameter(withAddress: 1)?.value = 0
        $0.parameter(withAddress: 2)?.value = 2
        $0.parameter(withAddress: 3)?.value = 1
    }
    check(abs(rms(bOut) - 0.5) < 0.01, "osc B alone set to Square and level 1")

    // Cutoff normalized 0.2 is about 80 Hz, well below A4 (440 Hz).
    let closedOut = try rendered { $0.parameter(withAddress: 4)?.value = 0.2 }
    check(rms(closedOut) < rms(defaultOut) * 0.2, "closing the filter cuts A4 hard")

    // Attack about 500 ms: the first 10 ms should be far quieter than the default.
    let slowAttackOut = try rendered { $0.parameter(withAddress: 6)?.value = 0.675 }
    let earlySlice = Array(slowAttackOut[0..<480])
    check(rms(earlySlice) < rms(defaultOut) * 0.2, "a slow attack fades the note in")

    let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "build/au_output.wav"
    try writeWAV(tone, to: out)
    print("  wrote \(out)")
} catch {
    print("  FAIL unexpected error: \(error)")
    failures += 1
}

print("\n\(failures == 0 ? "PASSED" : "FAILED") (\(failures) failure\(failures == 1 ? "" : "s"))")
exit(failures == 0 ? 0 : 1)
