// The AUv3 extension's principal class. Hosts such as GarageBand create the
// audio unit through it, and it shows the synth's SwiftUI panel as the plug-in
// window. The panel is the same UI the standalone app uses (ui/SynthUI.swift).
import AppKit
import AudioToolbox
import CoreAudioKit
import SwiftUI

@objc(SynthAudioUnitViewController)
public final class SynthAudioUnitViewController: AUViewController, AUAudioUnitFactory {
    private let model = SynthModel()
    private var unit: SynthAudioUnit?
    private var observerToken: AUParameterObserverToken?

    public override init(nibName nibNameOrNil: NSNib.Name?, bundle nibBundleOrNil: Bundle?) {
        super.init(nibName: nil, bundle: nil)
    }

    public required init?(coder: NSCoder) {
        super.init(coder: coder)
    }

    public override func loadView() {
        let size = NSSize(width: 980, height: 600)
        let hosting = NSHostingView(rootView: ContentView(model: model, computerKeyboard: false, showScope: false))
        hosting.frame = NSRect(origin: .zero, size: size)
        view = hosting
        preferredContentSize = size
    }

    public func createAudioUnit(with componentDescription: AudioComponentDescription) throws -> AUAudioUnit {
        let unit = try SynthAudioUnit(componentDescription: componentDescription, options: [])
        connect(unit)
        return unit
    }

    /// Wires the panel to the unit: UI writes go to the parameter tree, and
    /// changes made by the host (automation, presets) come back to the panel.
    private func connect(_ unit: SynthAudioUnit) {
        self.unit = unit
        guard let tree = unit.parameterTree else { return }
        for parameter in tree.allParameters {
            model.params[Int(parameter.address)] = parameter.value
        }
        model.sendParam = { index, value in
            tree.parameter(withAddress: AUParameterAddress(index))?.value = value
        }
        model.sendMIDI = { [weak unit] bytes in
            guard let block = unit?.scheduleMIDIEventBlock else { return }
            var b = bytes
            b.withUnsafeMutableBufferPointer { p in
                _ = block(AUEventSampleTimeImmediate, 0, p.count, p.baseAddress!)
            }
        }
        let model = self.model
        observerToken = tree.token(byAddingParameterObserver: { address, value in
            DispatchQueue.main.async {
                model.params[Int(address)] = value
            }
        })
    }
}
