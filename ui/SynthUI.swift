// Shared synth UI, used by the standalone app and the AU plug-in window.
import AppKit
import Combine
import SwiftUI

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

func noteName(_ midi: Int) -> String {
    let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return "\(names[midi % 12])\(midi / 12 - 1)"
}

// MARK: - Shared synth state

/// The synth's UI state. The standalone app and the AU plug-in window both
/// drive this model. Whoever owns the audio unit supplies how parameter and
/// MIDI writes reach it.
final class SynthModel: ObservableObject {
    @Published var status = ""
    /// Notes whose keys are currently down (mouse, computer keyboard or MIDI).
    @Published var heldNotes: Set<Int> = []
    @Published var velocity: Double = 100
    /// Parameter values, indexed by the core's parameter index (its AU address).
    @Published var params: [Float] = (0..<Int(synth_param_count())).map {
        synth_param_info(Int32($0)).defaultValue
    }
    let scope = ScopeBuffer()

    /// Writes a parameter to the audio unit.
    var sendParam: ((Int, Float) -> Void)?
    /// Sends raw MIDI bytes to the audio unit.
    var sendMIDI: (([UInt8]) -> Void)?

    func setParam(_ index: Int, _ value: Float) {
        params[index] = value
        sendParam?(index, value)
    }

    func noteOn(_ note: Int) {
        guard !heldNotes.contains(note) else { return }
        heldNotes.insert(note)
        sendMIDI?([0x90, UInt8(note), UInt8(velocity)])
    }

    /// Only releases the note if it is still the one we started, so a fast
    /// slide from one key to the next does not cut the new note.
    func noteOff(_ note: Int) {
        guard heldNotes.contains(note) else { return }
        heldNotes.remove(note)
        sendMIDI?([0x80, UInt8(note), 0])
    }
}

// MARK: - Theme

/// Monotone greys on a dark base. hi marks the sound sources and live values,
/// mid marks shaping controls (filter, envelopes, lfo, fx).
enum Theme {
    static let bg = Color(red: 0.07, green: 0.07, blue: 0.09)
    static let panel = Color(red: 0.11, green: 0.11, blue: 0.14)
    static let track = Color(red: 0.22, green: 0.22, blue: 0.26)
    static let text = Color(red: 0.90, green: 0.90, blue: 0.93)
    static let dim = Color(red: 0.55, green: 0.55, blue: 0.62)
    static let hi = Color(red: 0.86, green: 0.86, blue: 0.89)
    static let mid = Color(red: 0.62, green: 0.62, blue: 0.67)

    static func font(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }
}

/// A plain rounded panel with a small lowercase title.
struct Panel<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(Theme.font(11, .medium))
                .foregroundStyle(Theme.dim)
                .lineLimit(1)
            content
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: 12).fill(Theme.panel))
    }
}

/// Small waveform selector.
struct WaveChip: View {
    let name: String
    let selected: Bool
    let accent: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(name)
                .font(Theme.font(10, .medium))
                .lineLimit(1)
                .minimumScaleFactor(0.5)
                .foregroundStyle(selected ? Theme.bg : Theme.dim)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 4)
                .background(RoundedRectangle(cornerRadius: 6)
                    .fill(selected ? accent : Theme.track.opacity(0.5)))
        }
        .buttonStyle(.plain)
    }
}

/// Rotary knob for a normalized (0-1) value. Drag up or down to turn it.
struct KnobView: View {
    let title: String
    @Binding var value: Double
    let valueText: String
    let accent: Color
    var size: CGFloat = 46

    @State private var dragStart: Double?

    var body: some View {
        VStack(spacing: 5) {
            ZStack {
                // 270-degree sweep, from 7:30 to 4:30.
                Circle()
                    .trim(from: 0, to: 0.75)
                    .stroke(Theme.track, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .rotationEffect(.degrees(135))
                    .padding(4)
                Circle()
                    .trim(from: 0, to: 0.75 * value)
                    .stroke(accent, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .rotationEffect(.degrees(135))
                    .padding(4)
                Capsule()
                    .fill(Theme.text)
                    .frame(width: 2, height: size * 0.28)
                    .offset(y: -size * 0.22)
                    .rotationEffect(.degrees(-135 + 270 * value))
            }
            .frame(width: size, height: size)
            .contentShape(Circle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { g in
                        if dragStart == nil { dragStart = value }
                        // 200 points of vertical drag sweeps the whole range.
                        let v = (dragStart ?? value) - Double(g.translation.height) / 200
                        value = min(1, max(0, v))
                    }
                    .onEnded { _ in dragStart = nil }
            )

            Text(title)
                .font(Theme.font(10))
                .foregroundStyle(Theme.dim)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            Text(valueText)
                .font(Theme.font(10, .medium))
                .foregroundStyle(Theme.text)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity)
    }
}

/// Interactive mixer graph. Each source is a column whose height is its level.
/// Drag anywhere in the left half to set osc a, the right half for osc b.
struct MixerGraph: View {
    @Binding var a: Double
    @Binding var b: Double

    var body: some View {
        VStack(spacing: 6) {
            Text("a \(Int((a * 100).rounded()))   b \(Int((b * 100).rounded()))")
                .font(Theme.font(10, .medium))
                .foregroundStyle(Theme.text)
            GeometryReader { geo in
                let w = geo.size.width
                let h = geo.size.height
                Canvas { ctx, size in
                    // Horizontal grid at 25, 50 and 75 percent.
                    for i in 1...3 {
                        var grid = Path()
                        let y = size.height * CGFloat(i) / 4
                        grid.move(to: CGPoint(x: 0, y: y))
                        grid.addLine(to: CGPoint(x: size.width, y: y))
                        ctx.stroke(grid, with: .color(Theme.track.opacity(0.6)), lineWidth: 1)
                    }
                    let xs = [size.width * 0.25, size.width * 0.75]
                    let vals = [a, b]
                    var curve = Path()
                    for i in 0..<2 {
                        let top = size.height * CGFloat(1 - vals[i])
                        let bar = CGRect(x: xs[i] - 22, y: top, width: 44, height: size.height - top)
                        ctx.fill(Path(roundedRect: bar, cornerRadius: 6), with: .color(Theme.hi.opacity(0.25)))
                        ctx.stroke(Path(roundedRect: bar, cornerRadius: 6), with: .color(Theme.hi.opacity(0.6)), lineWidth: 1)
                        if i == 0 { curve.move(to: CGPoint(x: xs[i], y: top)) } else { curve.addLine(to: CGPoint(x: xs[i], y: top)) }
                    }
                    ctx.stroke(curve, with: .color(Theme.hi), style: StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: [4, 3]))
                    for i in 0..<2 {
                        let top = size.height * CGFloat(1 - vals[i])
                        let dot = CGRect(x: xs[i] - 6, y: top - 6, width: 12, height: 12)
                        ctx.fill(Path(ellipseIn: dot), with: .color(Theme.hi))
                    }
                }
                .frame(width: w, height: h)
                .background(Theme.panel)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { g in
                            let v = min(1, max(0, Double(1 - g.location.y / h)))
                            if g.location.x < w / 2 { a = v } else { b = v }
                        }
                )
            }
            .frame(height: 130)
            HStack {
                Text("osc a").frame(maxWidth: .infinity)
                Text("osc b").frame(maxWidth: .infinity)
            }
            .font(Theme.font(10))
            .foregroundStyle(Theme.dim)
        }
    }
}

struct ScopeView: View {
    let buffer: ScopeBuffer

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30)) { _ in
            let snap = buffer.snapshot()
            Canvas { ctx, size in
                var mid = Path()
                mid.move(to: CGPoint(x: 0, y: size.height / 2))
                mid.addLine(to: CGPoint(x: size.width, y: size.height / 2))
                ctx.stroke(mid, with: .color(Theme.track), lineWidth: 1)

                var wave = Path()
                let s = snap.samples
                for (i, v) in s.enumerated() {
                    let x = size.width * CGFloat(i) / CGFloat(s.count - 1)
                    let y = size.height / 2 - CGFloat(v) * size.height / 2 * 0.9
                    if i == 0 { wave.move(to: CGPoint(x: x, y: y)) } else { wave.addLine(to: CGPoint(x: x, y: y)) }
                }
                ctx.stroke(wave, with: .color(Theme.hi), style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
            }
            .background(Theme.panel)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(alignment: .topTrailing) {
                Text(String(format: "peak %.2f", snap.peak))
                    .font(Theme.font(10))
                    .foregroundStyle(Theme.dim)
                    .padding(8)
            }
        }
    }
}

struct KeyboardView: View {
    @ObservedObject var model: SynthModel

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
                HStack(spacing: 2) {
                    ForEach(whites, id: \.self) { note in
                        key(note, isBlack: false)
                            .frame(width: w - 2, height: geo.size.height)
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
        let pressed = model.heldNotes.contains(note)
        let fill: Color = pressed
            ? Theme.hi
            : (isBlack ? Color.black : Color(red: 0.84, green: 0.84, blue: 0.88))
        return RoundedRectangle(cornerRadius: 4)
            .fill(fill)
            .overlay(alignment: .bottom) {
                Text(noteName(note).lowercased())
                    .font(Theme.font(9))
                    .foregroundStyle(isBlack && !pressed ? Theme.dim : Color.black.opacity(0.5))
                    .padding(5)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        model.noteOn(note)
                    }
                    .onEnded { _ in model.noteOff(note) }
            )
    }
}

/// Formats a parameter's current value for display, by its index.
private func displayText(_ model: SynthModel, _ index: Int) -> String {
    let v = model.params[index]
    switch index {
    case Int(SYNTH_PARAM_FILTER_CUTOFF):
        let hz = synth_cutoff_hz(v)
        return hz >= 1000 ? String(format: "%.1fk", hz / 1000) : String(format: "%.0fhz", hz)
    case Int(SYNTH_PARAM_ENV_ATTACK), Int(SYNTH_PARAM_ENV_DECAY), Int(SYNTH_PARAM_ENV_RELEASE),
         Int(SYNTH_PARAM_FILTER_ENV_ATTACK), Int(SYNTH_PARAM_FILTER_ENV_DECAY), Int(SYNTH_PARAM_FILTER_ENV_RELEASE):
        let secs = synth_envelope_seconds(v)
        return secs < 1 ? String(format: "%.0fms", secs * 1000) : String(format: "%.2fs", secs)
    case Int(SYNTH_PARAM_LFO_RATE):
        return String(format: "%.2fhz", synth_lfo_hz(v))
    case Int(SYNTH_PARAM_OSC_A_DETUNE), Int(SYNTH_PARAM_OSC_B_DETUNE):
        return String(format: "%+.0fct", v)
    case Int(SYNTH_PARAM_LFO_PITCH_DEPTH):
        return String(format: "%.0fct", v)
    default:
        return "\(Int((v * 100).rounded()))%"
    }
}

/// Computer keyboard layout, one octave from C4, like a piano: white keys on
/// the home row, black keys above them. Several keys can be held for chords.
let computerKeyNotes: [String: Int] = [
    "a": 60, "w": 61, "s": 62, "e": 63, "d": 64, "f": 65,
    "t": 66, "g": 67, "y": 68, "h": 69, "u": 70, "j": 71, "k": 72,
]

struct ContentView: View {
    @ObservedObject var model: SynthModel
    /// Lets the computer keyboard play notes. The plug-in turns this off, since
    /// the host app already sends keys to the instrument.
    var computerKeyboard = true
    /// The standalone app shows the live scope; the plug-in window hides it.
    var showScope = true

    private var waveNames: [String] {
        (0..<Int(synth_waveform_count())).map { String(cString: synth_waveform_name(Int32($0))).lowercased() }
    }

    /// Two-way binding from a parameter index to a slider or knob.
    private func binding(_ index: Int) -> Binding<Double> {
        Binding(
            get: { Double(model.params[index]) },
            set: { model.setParam(index, Float($0)) })
    }

    private func knob(_ title: String, _ index: Int, _ accent: Color, size: CGFloat = 46) -> some View {
        KnobView(title: title,
                 value: binding(index),
                 valueText: displayText(model, index),
                 accent: accent,
                 size: size)
    }

    private func oscillator(_ title: String, wave: Int, detune: Int) -> some View {
        Panel(title: title) {
            HStack(spacing: 4) {
                ForEach(Array(waveNames.enumerated()), id: \.offset) { i, name in
                    WaveChip(name: name,
                             selected: Int(model.params[wave].rounded()) == i,
                             accent: Theme.hi) {
                        model.setParam(wave, Float(i))
                    }
                }
            }
            HStack {
                Spacer(minLength: 0)
                knob("detune", detune, Theme.hi)
                Spacer(minLength: 0)
            }
        }
    }

    var body: some View {
        VStack(spacing: 10) {
            header

            if showScope {
                ScopeView(buffer: model.scope)
                    .frame(height: 80)
            }

            HStack(spacing: 10) {
                oscillator("osc a", wave: Int(SYNTH_PARAM_OSC_A_WAVE), detune: Int(SYNTH_PARAM_OSC_A_DETUNE))
                oscillator("osc b", wave: Int(SYNTH_PARAM_OSC_B_WAVE), detune: Int(SYNTH_PARAM_OSC_B_DETUNE))

                Panel(title: "mixer") {
                    MixerGraph(a: binding(Int(SYNTH_PARAM_OSC_A_LEVEL)),
                               b: binding(Int(SYNTH_PARAM_OSC_B_LEVEL)))
                }

                Panel(title: "filter") {
                    HStack {
                        knob("cutoff", Int(SYNTH_PARAM_FILTER_CUTOFF), Theme.mid)
                        knob("res", Int(SYNTH_PARAM_FILTER_RESONANCE), Theme.mid)
                        knob("env", Int(SYNTH_PARAM_FILTER_ENV_AMOUNT), Theme.mid)
                    }
                }
            }

            HStack(alignment: .top, spacing: 10) {
                Panel(title: "amp envelope") {
                    HStack {
                        knob("a", Int(SYNTH_PARAM_ENV_ATTACK), Theme.mid, size: 40)
                        knob("d", Int(SYNTH_PARAM_ENV_DECAY), Theme.mid, size: 40)
                        knob("s", Int(SYNTH_PARAM_ENV_SUSTAIN), Theme.mid, size: 40)
                        knob("r", Int(SYNTH_PARAM_ENV_RELEASE), Theme.mid, size: 40)
                    }
                }

                Panel(title: "filter envelope") {
                    HStack {
                        knob("a", Int(SYNTH_PARAM_FILTER_ENV_ATTACK), Theme.mid, size: 40)
                        knob("d", Int(SYNTH_PARAM_FILTER_ENV_DECAY), Theme.mid, size: 40)
                        knob("s", Int(SYNTH_PARAM_FILTER_ENV_SUSTAIN), Theme.mid, size: 40)
                        knob("r", Int(SYNTH_PARAM_FILTER_ENV_RELEASE), Theme.mid, size: 40)
                    }
                }

                Panel(title: "lfo") {
                    HStack {
                        knob("rate", Int(SYNTH_PARAM_LFO_RATE), Theme.mid, size: 40)
                        knob("pitch", Int(SYNTH_PARAM_LFO_PITCH_DEPTH), Theme.mid, size: 40)
                        knob("filter", Int(SYNTH_PARAM_LFO_FILTER_DEPTH), Theme.mid, size: 40)
                    }
                }

                Panel(title: "fx") {
                    HStack {
                        knob("drive", Int(SYNTH_PARAM_DRIVE), Theme.mid, size: 40)
                        knob("verb", Int(SYNTH_PARAM_REVERB_MIX), Theme.mid, size: 40)
                        knob("size", Int(SYNTH_PARAM_REVERB_SIZE), Theme.mid, size: 40)
                    }
                }
            }

            KeyboardView(model: model)
                .frame(height: 90)
        }
        .padding(14)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg)
        .focusable()
        .onKeyPress(phases: [.down, .up]) { press in
            guard computerKeyboard, let note = computerKeyNotes[press.characters.lowercased()] else {
                return .ignored
            }
            if press.phase == .down {
                model.noteOn(note)
            } else {
                model.noteOff(note)
            }
            return .handled
        }
        .foregroundStyle(Theme.text)
        .frame(minWidth: 980, idealWidth: 980, minHeight: 600, idealHeight: 600)
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 16) {
            Text("synthrl")
                .font(Theme.font(16, .semibold))
                .foregroundStyle(Theme.text)
            Spacer(minLength: 8)
            HStack(spacing: 8) {
                Text("velocity")
                    .font(Theme.font(10))
                    .foregroundStyle(Theme.dim)
                Slider(value: $model.velocity, in: 1...127)
                    .tint(Theme.mid)
                    .frame(width: 120)
                Text("\(Int(model.velocity))")
                    .font(Theme.font(10, .medium))
                    .frame(width: 24, alignment: .trailing)
            }
            Text(model.heldNotes.isEmpty
                 ? "idle"
                 : "playing " + model.heldNotes.sorted().map { noteName($0).lowercased() }.joined(separator: " "))
                .lineLimit(1)
                .font(Theme.font(10))
                .foregroundStyle(Theme.dim)
                .frame(width: 80, alignment: .trailing)
            Text(model.status.lowercased())
                .font(Theme.font(10))
                .foregroundStyle(Theme.hi)
        }
    }
}

