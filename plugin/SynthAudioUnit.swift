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

    /// Builds the neural-source wavetables. Not real-time safe: call off the audio thread.
    func setTimbre(_ harmonics: [Float]) {
        harmonics.withUnsafeBufferPointer { buffer in
            synth_set_timbre(core, buffer.baseAddress, Int32(buffer.count))
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

/// Key under which the timbre is stored in AUAudioUnit.fullState.
let synthTimbreStateKey = "synthTimbre"

@objc(SynthAudioUnit)
public final class SynthAudioUnit: AUAudioUnit {
    private let kernel = SynthKernel()
    private var timbreValues: [Float] = []
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

    /// Loads a neural-source timbre (harmonic amplitudes, fundamental first) and
    /// sets Source Mix to 1 so it is heard. Call from the main thread, not the audio thread.
    public func loadTimbre(_ harmonics: [Float]) {
        timbreValues = harmonics
        kernel.setTimbre(harmonics)
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
            }
            return state
        }
        set {
            super.fullState = newValue
            let stored = (newValue?[synthTimbreStateKey] as? [Double]) ?? []
            timbreValues = stored.map { Float($0) }
            kernel.setTimbre(timbreValues)  // an empty list clears the timbre
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
