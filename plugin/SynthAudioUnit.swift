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

@objc(SynthAudioUnit)
public final class SynthAudioUnit: AUAudioUnit {
    private let kernel = SynthKernel()
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
