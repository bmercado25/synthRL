import AudioToolbox
import AVFoundation

// Component identity. Must match plugin/Info.plist (AudioComponents entry).
let synthComponentType: OSType = 0x61756D75     // 'aumu' (music device)
let synthComponentSubtype: OSType = 0x53796E31  // 'Syn1'
let synthComponentManufacturer: OSType = 0x53795252  // 'SyRR'

/// Owns the C++ core. Created with the unit, so parameters can be set before
/// playback starts. Parameter writes come from the UI thread and are atomic in
/// the core; note events and rendering come from the audio thread.
final class SynthKernel {
    private let core: OpaquePointer

    init() {
        core = synth_create(44_100)
    }

    deinit {
        synth_destroy(core)
    }

    func setSampleRate(_ sampleRate: Double) {
        synth_set_sample_rate(core, sampleRate)
    }

    func setParam(_ index: Int, _ value: Float) {
        synth_set_param(core, Int32(index), value)
    }

    /// Builds the neural-source wavetables, and either a loudness envelope or the
    /// source's movement, in one step. An empty envelope means full level. Movement
    /// replaces the envelope. Not real-time safe: call off the audio thread.
    func setTimbre(_ harmonics: [Float], attack: [Float] = [], envelope: [Float],
                   movement: TimbreMovement? = nil) {
        // Movement replaces the envelope, so the envelope is dropped when movement is present.
        let loudness = movement == nil ? envelope : []
        harmonics.withUnsafeBufferPointer { h in
            attack.withUnsafeBufferPointer { at in
                loudness.withUnsafeBufferPointer { e in
                    withMovementTrack(movement) { track in
                        synth_set_timbre_full(core, h.baseAddress, Int32(h.count),
                                              at.count > 0 ? at.baseAddress : nil, Int32(at.count),
                                              e.count > 0 ? e.baseAddress : nil, Int32(e.count),
                                              track)
                    }
                }
            }
        }
    }

    /// Passes a movement track to `body` as a pointer valid for the call (nil when there is none).
    private func withMovementTrack(_ movement: TimbreMovement?,
                                   _ body: (UnsafePointer<SynthMovement>?) -> Void) {
        guard let movement else {
            body(nil)
            return
        }
        movement.amp.withUnsafeBufferPointer { a in
            movement.pitchCents.withUnsafeBufferPointer { p in
                movement.brightnessOct.withUnsafeBufferPointer { b in
                    let track = SynthMovement(
                        frameCount: Int32(min(a.count, p.count, b.count)),
                        loopStart: Int32(movement.loopStart),
                        loopEnd: Int32(movement.loopEnd),
                        amp: a.baseAddress,
                        pitchCents: p.baseAddress,
                        brightnessOct: b.baseAddress)
                    withUnsafePointer(to: track) { body($0) }
                }
            }
        }
    }

    func handleMIDI(status: UInt8, data1: UInt8, data2: UInt8) {
        switch status & 0xF0 {
        case 0x90: synth_note_on(core, Int32(data1), Int32(data2))
        case 0x80: synth_note_off(core, Int32(data1))
        default: break
        }
    }

    func render(into out: UnsafeMutablePointer<Float>, frames: Int) {
        synth_render(core, out, Int32(frames))
    }
}

/// Builds the AUParameterTree from the core's schema (one declaration, in C++).
func makeSynthParameters() -> [AUParameter] {
    (0..<Int(synth_param_count())).map { index in
        let info = synth_param_info(Int32(index))
        let isWaveform = info.kind == SYNTH_PARAM_WAVEFORM
        let choices: [String]? = isWaveform
            ? (0..<Int(synth_waveform_count())).map { String(cString: synth_waveform_name(Int32($0))) }
            : nil
        let parameter = AUParameterTree.createParameter(
            withIdentifier: String(cString: info.id),
            name: String(cString: info.name),
            address: AUParameterAddress(index),
            min: info.min,
            max: info.max,
            unit: .generic,
            unitName: nil,
            flags: [.flag_IsReadable, .flag_IsWritable],
            valueStrings: choices,
            dependentParameters: nil)
        parameter.value = info.defaultValue
        return parameter
    }
}

/// Keys under which the timbre, its loudness envelope and its movement are stored in AUAudioUnit.fullState.
let synthTimbreStateKey = "synthTimbre"
let synthTimbreEnvelopeStateKey = "synthTimbreEnvelope"
let synthTimbreMovementStateKey = "synthTimbreMovement"
let synthTimbreAttackStateKey = "synthTimbreAttack"

/// The source's movement (step 2c): loudness, pitch and brightness tracks on a
/// 50 Hz grid, and the loop region in frames. Same layout as the "movement" block
/// of a timbre file.
public struct TimbreMovement {
    public var loopStart: Int
    public var loopEnd: Int
    public var amp: [Float]
    public var pitchCents: [Float]
    public var brightnessOct: [Float]

    public init(loopStart: Int, loopEnd: Int, amp: [Float], pitchCents: [Float], brightnessOct: [Float]) {
        self.loopStart = loopStart
        self.loopEnd = loopEnd
        self.amp = amp
        self.pitchCents = pitchCents
        self.brightnessOct = brightnessOct
    }

    /// Property-list form, for AUAudioUnit.fullState (and so for presets).
    var stateValue: [String: Any] {
        ["loopStart": Double(loopStart), "loopEnd": Double(loopEnd),
         "amp": amp.map { Double($0) }, "pitchCents": pitchCents.map { Double($0) },
         "brightnessOct": brightnessOct.map { Double($0) }]
    }

    init?(stateValue: [String: Any]) {
        guard let start = stateValue["loopStart"] as? Double,
              let end = stateValue["loopEnd"] as? Double,
              let amp = stateValue["amp"] as? [Double],
              let pitch = stateValue["pitchCents"] as? [Double],
              let bright = stateValue["brightnessOct"] as? [Double] else { return nil }
        self.init(loopStart: Int(start), loopEnd: Int(end),
                  amp: amp.map { Float($0) }, pitchCents: pitch.map { Float($0) },
                  brightnessOct: bright.map { Float($0) })
    }
}

@objc(SynthAudioUnit)
public final class SynthAudioUnit: AUAudioUnit {
    private let kernel = SynthKernel()
    private var timbreValues: [Float] = []
    private var timbreEnvelope: [Float] = []
    private var timbreAttack: [Float] = []
    private var timbreMovement: TimbreMovement?
    private var outputBus: AUAudioUnitBus!
    private var outputBusArray: AUAudioUnitBusArray!

    public override init(componentDescription: AudioComponentDescription,
                         options: AudioComponentInstantiationOptions = []) throws {
        try super.init(componentDescription: componentDescription, options: options)
        let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 2)!
        outputBus = try AUAudioUnitBus(format: format)
        outputBusArray = AUAudioUnitBusArray(audioUnit: self, busType: .output, busses: [outputBus])
        maximumFramesToRender = 4096

        let parameters = makeSynthParameters()
        let tree = AUParameterTree.createTree(withChildren: parameters)
        let kernel = self.kernel
        tree.implementorValueObserver = { parameter, value in
            kernel.setParam(Int(parameter.address), value)
        }
        parameterTree = tree
        for parameter in parameters {
            kernel.setParam(Int(parameter.address), parameter.value)
        }
    }

    /// Loads a neural-source timbre (harmonic amplitudes, fundamental first), with an
    /// optional loudness envelope over the note or the source's movement, and sets
    /// Source Mix to 1 so it is heard. Call from the main thread, not the audio thread.
    public func loadTimbre(_ harmonics: [Float], attack: [Float] = [], envelope: [Float] = [],
                           movement: TimbreMovement? = nil) {
        timbreValues = harmonics
        timbreAttack = attack
        timbreEnvelope = envelope
        timbreMovement = movement
        kernel.setTimbre(harmonics, attack: attack, envelope: envelope, movement: movement)
        let address = AUParameterAddress(Int(SYNTH_PARAM_SOURCE_MIX))
        parameterTree?.parameter(withAddress: address)?.value = 1
    }

    /// The timbre travels in the unit's full state, the standard AUv3 channel
    /// for settings. Hosts set it through AUAudioUnit, so they never need to
    /// cast to this class (the host's copy and the extension's copy differ).
    /// It is also saved with presets.
    public override var fullState: [String: Any]? {
        get {
            var state = super.fullState ?? [:]
            if !timbreValues.isEmpty {
                state[synthTimbreStateKey] = timbreValues.map { Double($0) }
                if !timbreAttack.isEmpty {
                    state[synthTimbreAttackStateKey] = timbreAttack.map { Double($0) }
                }
                if !timbreEnvelope.isEmpty {
                    state[synthTimbreEnvelopeStateKey] = timbreEnvelope.map { Double($0) }
                }
                if let movement = timbreMovement {
                    state[synthTimbreMovementStateKey] = movement.stateValue
                }
            }
            return state
        }
        set {
            super.fullState = newValue
            let stored = (newValue?[synthTimbreStateKey] as? [Double]) ?? []
            let storedEnvelope = (newValue?[synthTimbreEnvelopeStateKey] as? [Double]) ?? []
            let storedMovement = (newValue?[synthTimbreMovementStateKey] as? [String: Any])
                .flatMap(TimbreMovement.init(stateValue:))
            let storedAttack = (newValue?[synthTimbreAttackStateKey] as? [Double]) ?? []
            timbreValues = stored.map { Float($0) }
            timbreAttack = storedAttack.map { Float($0) }
            timbreEnvelope = storedEnvelope.map { Float($0) }
            timbreMovement = storedMovement
            // An empty harmonic list clears the timbre (with its attack, envelope and movement).
            kernel.setTimbre(timbreValues,
                             attack: timbreValues.isEmpty ? [] : timbreAttack,
                             envelope: timbreValues.isEmpty ? [] : timbreEnvelope,
                             movement: timbreValues.isEmpty ? nil : timbreMovement)
        }
    }

    public override var outputBusses: AUAudioUnitBusArray { outputBusArray }

    public override var canProcessInPlace: Bool { false }

    public override func allocateRenderResources() throws {
        try super.allocateRenderResources()
        kernel.setSampleRate(outputBus.format.sampleRate)
    }

    public override var internalRenderBlock: AUInternalRenderBlock {
        let kernel = self.kernel
        return { _, _, frameCount, _, outputData, realtimeEventListHead, _ in
            // Apply MIDI events. Sample-accurate scheduling comes later.
            var event = realtimeEventListHead
            while let e = event {
                if e.pointee.head.eventType == .MIDI {
                    let midi = e.pointee.MIDI
                    kernel.handleMIDI(status: midi.data.0,
                                      data1: midi.data.1,
                                      data2: midi.data.2)
                }
                event = UnsafePointer(e.pointee.head.next)
            }

            // Render the mono core, then copy it to every output channel.
            let frames = Int(frameCount)
            let buffers = UnsafeMutableAudioBufferListPointer(outputData)
            guard let first = buffers.first?.mData?.assumingMemoryBound(to: Float.self) else {
                return kAudioUnitErr_NoConnection
            }
            kernel.render(into: first, frames: frames)
            for buffer in buffers.dropFirst() {
                if let dst = buffer.mData?.assumingMemoryBound(to: Float.self) {
                    dst.update(from: first, count: frames)
                }
            }
            return noErr
        }
    }
}
