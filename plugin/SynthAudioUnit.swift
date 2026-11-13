import AudioToolbox
import AVFoundation

// Component identity. Must match plugin/Info.plist (AudioComponents entry).
let synthComponentType: OSType = 0x61756D75     // 'aumu' (music device)
let synthComponentSubtype: OSType = 0x53796E31  // 'Syn1'
let synthComponentManufacturer: OSType = 0x53795252  // 'SyRR'

/// Owns the C++ core. Lives on the audio thread after allocateRenderResources.
final class SynthKernel {
    private(set) var core: OpaquePointer?

    func start(sampleRate: Double) {
        stop()
        core = synth_create(sampleRate)
    }

    func stop() {
        if let core { synth_destroy(core) }
        core = nil
    }

    func handleMIDI(status: UInt8, data1: UInt8, data2: UInt8) {
        guard let core else { return }
        switch status & 0xF0 {
        case 0x90: synth_note_on(core, Int32(data1), Int32(data2))
        case 0x80: synth_note_off(core, Int32(data1))
        default: break
        }
    }

    func render(into out: UnsafeMutablePointer<Float>, frames: Int) {
        guard let core else {
            out.update(repeating: 0, count: frames)
            return
        }
        synth_render(core, out, Int32(frames))
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
    }

    public override var outputBusses: AUAudioUnitBusArray { outputBusArray }

    public override var canProcessInPlace: Bool { false }

    public override func allocateRenderResources() throws {
        try super.allocateRenderResources()
        kernel.start(sampleRate: outputBus.format.sampleRate)
    }

    public override func deallocateRenderResources() {
        kernel.stop()
        super.deallocateRenderResources()
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
