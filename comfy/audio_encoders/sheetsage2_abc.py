"""SheetSage2 timed events to two-voice ABC, adapted from m-a-p/SheetSage2."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Sequence

import numpy as np


_SUPPORTED_DURATION_UNITS = frozenset({1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48})
_MUSIC_ELEMENT_RE = re.compile(
    r'"(?P<quoted>[^"]*)"'
    r"|\[K:(?P<key>[^\]]+)\]"
    r"|(?P<note>[_=^]*[A-Ga-gz][,']*)(?P<duration>\d*)(?P<tie>-?)"
)


def interval_rows(events, field, duration):
    rows = [[event["time"], duration, event["values"][field]] for event in events if field in event["values"]]
    for previous, current in zip(rows, rows[1:]):
        previous[1] = current[0]
    return [row for row in rows if row[1] > row[0]]


def events_to_abc(events, duration, melody_only=True):
    beats, meter = [], None
    notes = {"Vocal": [], "Ins": []}
    for event in events:
        rhythm = event["values"].get("rhythm", {})
        meter = rhythm.get("meter", meter)
        eighth = rhythm.get("eighth_position")
        if eighth is not None and meter is not None:
            position = Fraction(eighth * meter[1], 8)
            if position.denominator != 1 or not 0 <= position < meter[0]:
                raise BeatGridError(f"Beat position {eighth} is outside the decoded {meter[0]}/{meter[1]} grid.")
            beats.append(BeatEvent(event["time"], int(position) + 1, meter[0], meter[1]))
        for note in event["values"].get("melody", ()):
            end = min(duration, note["end_time"])
            if end > event["time"]:
                notes[VOICE_IDS[note["track"]]].append([event["time"], end, note["pitch"]])
    if len(beats) < 2:
        raise BeatGridError("SheetSage2 needs at least two decoded beats to produce ABC.")
    if any(current.time <= previous.time for previous, current in zip(beats, beats[1:])):
        raise BeatGridError("SheetSage2 decoded beat times are not increasing.")
    period = float(np.median(np.diff([beat.time for beat in beats[-9:]])))
    while beats[-1].time < duration - 1e-6:
        previous = beats[-1]
        beats.append(BeatEvent(previous.time + period, previous.beat_id % previous.declared_numerator + 1,
                               previous.declared_numerator, previous.denominator))
    intervals = {}
    for field in ("key", "structure", "chord"):
        intervals[field] = [[max(beats[0].time, start), min(beats[-1].time, end), value]
                            for start, end, value in interval_rows(events, field, duration)
                            if end > beats[0].time and start < beats[-1].time]
    if not intervals["key"]:
        raise AbcRebuildError("SheetSage2 did not decode a key for the ABC score.")
    keys = [(start, end, key_symbol_to_abc(key)) for start, end, key in intervals["key"]]
    measures, diagnostics = infer_measures(beats)
    times, quarters, denominators = _build_grid(beats, measures)
    voices = {}
    for voice, track in notes.items():
        track.sort(key=lambda note: (note[0], note[2], note[1]))
        for previous, current in zip(track, track[1:]):
            if previous[1] > current[0] + 1e-6:
                previous[1] = current[0]
        voices[voice] = _notes_to_arr([note for note in track if note[1] > note[0] + 1e-6], times, voice)
    score = RebuiltAbcScore(
        beats=beats, measures=measures, subbeat_times=times, subbeat_quarters=quarters,
        subbeat_denominators=denominators,
        key_arr=_fill_intervals(keys, times, default=keys[0][2], dtype="<U16"),
        chord_arr=np.full(len(times), "N", dtype="<U64") if melody_only else _fill_intervals(intervals["chord"], times, default="N", dtype="<U64"),
        structure_events=_structure_events(intervals["structure"], times), voice_arrs=voices, diagnostics=diagnostics,
    )
    return score_to_abc(score)


SUBBEAT_DIVISION = 4

VOICE_IDS = ("Vocal", "Ins")

NO_CHORDS = frozenset({"N", "X", "?"})

class AbcRebuildError(ValueError):
    """Base class for deterministic reconstruction failures."""

class BeatGridError(AbcRebuildError):
    pass

class ChordSymbolError(AbcRebuildError):
    pass

class MelodyVoiceError(AbcRebuildError):
    pass

@dataclass(frozen=True)
class BeatEvent:
    time: float
    beat_id: int
    declared_numerator: int
    denominator: int

@dataclass(frozen=True)
class Measure:
    index: int
    start_beat: int
    end_beat: int
    numerator: int
    denominator: int
    pickup: bool = False
    partial: bool = False
    inferred: bool = False
    notated_numerator: int | None = None
    notated_denominator: int | None = None
    pad_before: bool = False

    @property
    def beat_count(self) -> int:
        return self.end_beat - self.start_beat

    @property
    def start_t(self) -> int:
        return self.start_beat * SUBBEAT_DIVISION

    @property
    def end_t(self) -> int:
        return self.end_beat * SUBBEAT_DIVISION

    @property
    def abc_numerator(self) -> int:
        return self.notated_numerator or self.numerator

    @property
    def abc_denominator(self) -> int:
        return self.notated_denominator or self.denominator

@dataclass
class RebuiltAbcScore:
    beats: list[BeatEvent]
    measures: list[Measure]
    subbeat_times: np.ndarray
    subbeat_quarters: np.ndarray
    subbeat_denominators: np.ndarray
    key_arr: np.ndarray
    chord_arr: np.ndarray
    structure_events: list[tuple[int, str]]
    voice_arrs: dict[str, np.ndarray]
    diagnostics: list[str]
    subbeat_div: int = SUBBEAT_DIVISION

@dataclass
class MeasureGroup:
    measures: list[Measure]
    structure_labels: list[str]
    meter_changed: bool
    key_changed: bool

_QUALITY_TO_ABC = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "sus4": "sus4",
    "sus2": "sus2",
    "maj6": "6",
    "min6": "m6",
    "sus4(b7)": "7sus4",
    # abc2midi and SymMusic both accept the parenthesized major seventh.
    # Common aliases such as mmaj7/mM7 trigger abc2midi diagnostics.
    "minmaj7": "m(maj7)",
}

_NATURAL_PITCH_CLASS = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_LETTERS = "CDEFGAB"

_SHARP_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_FLAT_PITCH_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

_ROOT_RE = re.compile(r"^(?P<letter>[A-G])(?P<accidental>#{0,2}|b{0,2})$")

_BASS_DEGREE_RE = re.compile(r"^(?P<accidental>#{0,2}|b{0,2})(?P<degree>[1-9]|1[0-3])$")

_KEY_SIGNATURE_ACCIDENTALS = {
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
    "F#": 6,
    "C#": 7,
    "F": -1,
    "Bb": -2,
    "Eb": -3,
    "Ab": -4,
    "Db": -5,
    "Gb": -6,
    "Cb": -7,
    "Am": 0,
    "Em": 1,
    "Bm": 2,
    "F#m": 3,
    "C#m": 4,
    "G#m": 5,
    "D#m": 6,
    "A#m": 7,
    "Dm": -1,
    "Gm": -2,
    "Cm": -3,
    "Fm": -4,
    "Bbm": -5,
    "Ebm": -6,
    "Abm": -7,
}

_KEY_RELATIVE_PITCH_NAMES = {
    7: ("B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B"),
    6: ("B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B"),
    5: ("B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"),
    4: ("B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"),
    3: ("B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"),
    2: ("C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"),
    1: ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"),
    0: ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"),
    -1: ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"),
    -2: ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"),
    -3: ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"),
    -4: ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"),
    -5: ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"),
    -6: ("C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"),
    -7: ("C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb"),
}

def _mode_with_first_tiebreak(values: Sequence[int]) -> int:
    counts = Counter(values)
    maximum = max(counts.values())
    return next(value for value in values if counts[value] == maximum)

def infer_measures(beats: Sequence[BeatEvent]) -> tuple[list[Measure], list[str]]:
    """Infer self-consistent measures from actual downbeat boundaries."""

    downbeat_indices = [index for index, beat in enumerate(beats) if beat.beat_id == 1]
    if not downbeat_indices:
        raise BeatGridError("No downbeat (beat ID 1) exists in the beat lab")
    spans: list[tuple[int, int, bool, bool]] = []
    if downbeat_indices[0] > 0:
        spans.append((0, downbeat_indices[0], True, False))
    spans.extend(
        (start, end, False, False)
        for start, end in zip(downbeat_indices, downbeat_indices[1:])
    )
    if downbeat_indices[-1] < len(beats) - 1:
        # Exported beat labs use their last row as the score end boundary.  If
        # that row is not a downbeat, the final bar is intentionally truncated.
        spans.append((downbeat_indices[-1], len(beats) - 1, False, True))
    if not spans:
        raise BeatGridError("No positive-length measure exists between downbeats")

    diagnostics = []
    measures = []
    for measure_index, (start, end, pickup, partial) in enumerate(spans):
        events = list(beats[start:end])
        # Downbeat spans still define measures when the model skips a beat ID.
        beat_count = len(events)

        denominators = [event.denominator for event in events]
        denominator = _mode_with_first_tiebreak(denominators)
        declared_numerators = [event.declared_numerator for event in events]
        declared_numerator = _mode_with_first_tiebreak(declared_numerators)
        numerator_conflict = any(value != beat_count for value in declared_numerators)
        denominator_conflict = any(value != denominator for value in denominators)
        pad_final_partial = (
            partial
            and len(set(declared_numerators)) == 1
            and not denominator_conflict
            and declared_numerator >= beat_count
        )
        inferred = pickup or partial or numerator_conflict or denominator_conflict
        if pad_final_partial and declared_numerator > beat_count:
            diagnostics.append(
                f"measure {measure_index}: padded final {beat_count}/{denominator} span "
                f"to declared {declared_numerator}/{denominator} with trailing rest"
            )
        elif numerator_conflict:
            diagnostics.append(
                f"measure {measure_index}: inferred {beat_count}/{denominator} from downbeat span; "
                f"declared numerators were {declared_numerators}"
            )
        if denominator_conflict:
            diagnostics.append(
                f"measure {measure_index}: placed denominator {denominator} at the measure boundary; "
                f"row declarations were {denominators}"
            )
        measures.append(
            Measure(
                index=measure_index,
                start_beat=start,
                end_beat=end,
                numerator=beat_count,
                denominator=denominator,
                pickup=pickup,
                partial=partial,
                inferred=inferred,
                notated_numerator=(
                    declared_numerator if pad_final_partial else beat_count
                ),
            )
        )
    if len(measures) >= 2:
        first = measures[0]
        following = measures[1]
        first_duration = first.numerator / first.denominator
        following_duration = (
            following.abc_numerator / following.abc_denominator
        )
        if first_duration < following_duration:
            measures[0] = replace(
                first,
                inferred=True,
                notated_numerator=following.abc_numerator,
                notated_denominator=following.abc_denominator,
                pad_before=True,
            )
            diagnostics.append(
                f"measure 0: padded leading {first.numerator}/{first.denominator} span "
                f"to {following.abc_numerator}/{following.abc_denominator} "
                f"with preceding rest"
            )
    return measures, diagnostics

def _build_grid(beats: Sequence[BeatEvent], measures: Sequence[Measure]):
    interval_denominators = np.zeros(len(beats) - 1, dtype=np.int32)
    for measure in measures:
        interval_denominators[measure.start_beat:measure.end_beat] = measure.denominator
    if np.any(interval_denominators == 0):
        raise BeatGridError("Downbeat spans do not cover every beat interval")

    subbeat_times = []
    subbeat_denominators = []
    quarter_positions = [0.0]
    current_quarter = 0.0
    for index in range(len(beats) - 1):
        start = beats[index].time
        end = beats[index + 1].time
        denominator = int(interval_denominators[index])
        times = np.linspace(start, end, SUBBEAT_DIVISION + 1)[:-1]
        subbeat_times.extend(float(value) for value in times)
        subbeat_denominators.extend([denominator] * SUBBEAT_DIVISION)
        quarter_step = 4.0 / denominator / SUBBEAT_DIVISION
        for _ in range(SUBBEAT_DIVISION):
            current_quarter += quarter_step
            quarter_positions.append(current_quarter)
    subbeat_times.append(beats[-1].time)
    subbeat_denominators.append(int(interval_denominators[-1]))
    return (
        np.asarray(subbeat_times, dtype=np.float64),
        np.asarray(quarter_positions, dtype=np.float64),
        np.asarray(subbeat_denominators, dtype=np.int32),
    )

def _subbeat_boundaries(subbeat_times: np.ndarray) -> np.ndarray:
    return (subbeat_times[:-1] + subbeat_times[1:]) / 2

def _quantize_time(time: float, subbeat_times: np.ndarray) -> int:
    return int(np.searchsorted(_subbeat_boundaries(subbeat_times), float(time)))

def _fill_intervals(rows, subbeat_times, *, default, dtype):
    result = np.full(len(subbeat_times), default, dtype=dtype)
    for start, end, value in rows:
        start_t = _quantize_time(start, subbeat_times)
        end_t = _quantize_time(end, subbeat_times)
        start_t = max(0, min(start_t, len(result) - 1))
        end_t = max(0, min(end_t, len(result) - 1))
        if start_t == end_t == len(result) - 1:
            continue
        if end_t <= start_t:
            raise AbcRebuildError(
                f"Interval {start:.6f}-{end:.6f} ({value}) is shorter than the ABC subbeat grid"
            )
        result[start_t:end_t] = value
    if len(result) > 1:
        result[-1] = result[-2]
    return result

def _structure_events(rows, subbeat_times):
    events = []
    for start, _, label in rows:
        t = _quantize_time(start, subbeat_times)
        t = max(0, min(t, len(subbeat_times) - 1))
        events.append((t, label))
    return events

def _notes_to_arr(notes, subbeat_times, voice_id):
    result = np.zeros(len(subbeat_times), dtype=np.int32)
    boundaries = _subbeat_boundaries(subbeat_times)
    for note in sorted(notes, key=lambda item: (item[0], item[1], item[2])):
        start_t = int(np.searchsorted(boundaries, note[0]))
        end_t = int(np.searchsorted(boundaries, note[1]))
        start_t = max(0, min(start_t, len(result) - 1))
        end_t = max(0, min(end_t, len(result) - 1))
        if start_t == end_t == len(result) - 1:
            continue
        if end_t <= start_t:
            raise MelodyVoiceError(
                f"{voice_id}: note pitch={note[2]} at {note[0]:.6f}-{note[1]:.6f} "
                "cannot be represented on the decoded subbeat grid"
            )
        if np.any(result[start_t:end_t] != 0):
            raise MelodyVoiceError(
                f"{voice_id}: overlapping quantized melody notes at subbeats {start_t}:{end_t}"
            )
        sustain = note[2] * 2 + 2
        result[start_t:end_t] = sustain
        result[start_t] = sustain + 1
    return result

def _pitch_class(root: str) -> tuple[int, str, str]:
    match = _ROOT_RE.fullmatch(root)
    if match is None:
        raise ChordSymbolError(f"Invalid pitch spelling {root!r}")
    letter = match.group("letter")
    accidental = match.group("accidental")
    offset = accidental.count("#") - accidental.count("b")
    return (_NATURAL_PITCH_CLASS[letter] + offset) % 12, letter, accidental

def portable_pitch_name(root: str, *, preserve_double: bool = False) -> str:
    pitch_class, _, accidental = _pitch_class(root)
    if preserve_double or len(accidental) <= 1:
        return root
    names = _SHARP_PITCH_NAMES if accidental.startswith("#") else _FLAT_PITCH_NAMES
    return names[pitch_class]

def _bass_degree_to_pitch(root: str, degree_text: str) -> str:
    if _ROOT_RE.fullmatch(degree_text):
        return portable_pitch_name(degree_text, preserve_double=True)
    match = _BASS_DEGREE_RE.fullmatch(degree_text)
    if match is None:
        raise ChordSymbolError(f"Invalid chord bass degree {degree_text!r}")
    root_pc, root_letter, root_accidental = _pitch_class(root)
    degree = int(match.group("degree"))
    degree_accidental = match.group("accidental")
    scale_semitones = (0, 2, 4, 5, 7, 9, 11)
    interval = scale_semitones[(degree - 1) % 7] + 12 * ((degree - 1) // 7)
    interval += degree_accidental.count("#") - degree_accidental.count("b")
    target_pc = (root_pc + interval) % 12

    target_letter_index = (_LETTERS.index(root_letter) + degree - 1) % 7
    target_letter = _LETTERS[target_letter_index]
    natural_pc = _NATURAL_PITCH_CLASS[target_letter]
    difference = (target_pc - natural_pc + 6) % 12 - 6
    if difference in {-2, -1, 0, 1, 2}:
        accidental = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[difference]
        return target_letter + accidental
    names = _SHARP_PITCH_NAMES if "#" in (root_accidental + degree_accidental) else _FLAT_PITCH_NAMES
    return names[target_pc]

def chord_symbol_to_abc(chord: str) -> str | None:
    chord = chord.strip()
    if chord in NO_CHORDS:
        return None
    if ":" not in chord:
        raise ChordSymbolError(f"Chord {chord!r} is missing the ':' quality separator")
    root, descriptor = chord.split(":", 1)
    if "/" in descriptor:
        quality, bass_degree = descriptor.split("/", 1)
    else:
        quality, bass_degree = descriptor, None
    if quality not in _QUALITY_TO_ABC:
        raise ChordSymbolError(
            f"Unsupported chord quality {quality!r} in {chord!r}; refusing to rewrite it as major"
        )
    chord_root = portable_pitch_name(root, preserve_double=True)
    text = chord_root + _QUALITY_TO_ABC[quality]
    if bass_degree:
        text += "/" + _bass_degree_to_pitch(root, bass_degree)
    return text

def key_symbol_to_abc(key: str) -> str:
    key = key.strip()
    if ":" in key:
        root, mode = key.split(":", 1)
        if mode not in {"major", "minor"}:
            raise AbcRebuildError(f"Unsupported key mode {mode!r} in {key!r}")
    elif key.endswith("m"):
        root, mode = key[:-1], "minor"
    else:
        root, mode = key, "major"
    root_pc, _, accidental = _pitch_class(root)
    candidate = portable_pitch_name(root) + ("m" if mode == "minor" else "")
    if candidate in _KEY_SIGNATURE_ACCIDENTALS:
        return candidate
    names = _FLAT_PITCH_NAMES if "b" in accidental else _SHARP_PITCH_NAMES
    candidate = names[root_pc] + ("m" if mode == "minor" else "")
    if candidate not in _KEY_SIGNATURE_ACCIDENTALS:
        fallback_names = _SHARP_PITCH_NAMES if names is _FLAT_PITCH_NAMES else _FLAT_PITCH_NAMES
        candidate = fallback_names[root_pc] + ("m" if mode == "minor" else "")
    if candidate not in _KEY_SIGNATURE_ACCIDENTALS:
        raise AbcRebuildError(f"Cannot encode portable ABC key for {key!r}")
    return candidate

def get_key_accidentals(key: str) -> list[int]:
    try:
        count = _KEY_SIGNATURE_ACCIDENTALS[key]
    except KeyError as exc:
        raise AbcRebuildError(f"Unsupported ABC key signature {key!r}") from exc
    accidentals = [0] * 7
    order = "FCGDAEB" if count > 0 else "BEADGCF"
    for letter in order[:abs(count)]:
        accidentals[_LETTERS.index(letter)] = 1 if count > 0 else -1
    return accidentals

def note_to_abc(note: int, key_accidentals: Sequence[int], measure_accidentals: dict) -> str:
    """Use key-relative spelling and write only bar-state changes.

    The two target parsers propagate an accidental to the same note letter in
    every octave until the next barline. ``measure_accidentals`` is therefore
    keyed by letter and reset by the caller for every bar (and after an inline
    key change). This preserves pitches across parsers while still omitting
    repeated accidental marks.  The key-relative spelling can use double
    accidentals in remote keys; MIDI G is F## in G# minor, for example.
    """

    accidental_count = sum(key_accidentals)
    try:
        pitch_name = _KEY_RELATIVE_PITCH_NAMES[accidental_count][note % 12]
    except KeyError as exc:
        raise AbcRebuildError(
            f"Unsupported key signature accidental count {accidental_count}"
        ) from exc
    letter = pitch_name[0]
    accidental = pitch_name[1:]
    accidental_number = {"": 0, "#": 1, "##": 2, "b": -1, "bb": -2}[accidental]
    octave = (note - 60) // 12
    # Cb and B# cross the MIDI octave boundary even though their written note
    # letter does not.
    if note % 12 == 11 and accidental_number == -1:
        octave += 1
    elif note % 12 == 0 and accidental_number == 1:
        octave -= 1
    scale_index = _LETTERS.index(letter)
    current_accidental = measure_accidentals.get(
        scale_index,
        key_accidentals[scale_index],
    )
    accidental_text = ""
    if current_accidental != accidental_number:
        measure_accidentals[scale_index] = accidental_number
        accidental_text = {-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"}[
            accidental_number
        ]

    if octave > 0:
        letter = letter.lower()
        if octave > 1:
            letter += "'" * (octave - 1)
    elif octave < 0:
        letter += "," * abs(octave)
    return accidental_text + letter

def abc_unit_denominator(score: RebuiltAbcScore) -> int:
    values = [
        denominator * score.subbeat_div
        for measure in score.measures
        for denominator in (measure.denominator, measure.abc_denominator)
    ]
    denominator = math.lcm(*values)
    if denominator > 1024:
        raise AbcRebuildError(f"Required ABC unit length 1/{denominator} is unreasonably small")
    return denominator

def _measure_actual_units(measure: Measure, unit_denominator: int) -> int:
    return measure.numerator * unit_denominator // measure.denominator

def _measure_abc_units(measure: Measure, unit_denominator: int) -> int:
    return measure.abc_numerator * unit_denominator // measure.abc_denominator

def _measure_padding_units(measure: Measure, unit_denominator: int) -> int:
    return (
        _measure_abc_units(measure, unit_denominator)
        - _measure_actual_units(measure, unit_denominator)
    )

def _duration_units(score: RebuiltAbcScore, start_t: int, end_t: int, unit_denominator: int) -> int:
    units = 0
    for denominator in score.subbeat_denominators[start_t:end_t]:
        divisor = int(denominator) * score.subbeat_div
        if unit_denominator % divisor:
            raise AbcRebuildError(
                f"ABC L:1/{unit_denominator} cannot express a 1/{divisor} subbeat exactly"
            )
        units += unit_denominator // divisor
    return units

def estimate_tempo(score: RebuiltAbcScore) -> float:
    seconds = score.subbeat_times[-1] - score.subbeat_times[0]
    quarter_notes = score.subbeat_quarters[-1] - score.subbeat_quarters[0]
    if seconds <= 0 or quarter_notes <= 0:
        raise AbcRebuildError("Cannot estimate tempo from a zero-duration score")
    return float(quarter_notes / seconds * 60.0)

def _continues_pitch(value: int, next_value: int) -> bool:
    if value <= 0:
        return False
    pitch = value // 2 - 1
    return next_value == pitch * 2 + 2

def _same_note_segment(value: int, next_value: int) -> bool:
    if value == 0:
        return next_value == 0
    pitch = value // 2 - 1
    return next_value == pitch * 2 + 2

def _split_duration_units(duration: int) -> list[int]:
    """Split a duration into values accepted by strict music parsers."""

    if duration <= 0:
        raise AbcRebuildError(f"Cannot serialize non-positive duration {duration}")
    result = []
    remaining = int(duration)
    while remaining:
        if remaining in _SUPPORTED_DURATION_UNITS:
            result.append(remaining)
            break
        candidates = [
            value
            for value in _SUPPORTED_DURATION_UNITS
            if value < remaining
        ]
        if not candidates:
            raise AbcRebuildError(
                f"Duration {duration} cannot be split into representable ABC values"
            )
        chunk = max(candidates)
        result.append(chunk)
        remaining -= chunk
    return result

def _duration_text(duration: int) -> str:
    return "" if duration == 1 else str(duration)

def _render_duration_tokens(
    prefix: str,
    note_text: str,
    duration: int,
    *,
    tie_out: bool,
) -> list[str]:
    chunks = _split_duration_units(duration)
    tokens = []
    for index, chunk in enumerate(chunks):
        continues = note_text != "z" and (
            index + 1 < len(chunks) or tie_out
        )
        tokens.append(
            (prefix if index == 0 else "")
            + note_text
            + _duration_text(chunk)
            + ("-" if continues else "")
        )
    return tokens

def _render_voice_measure(
    score: RebuiltAbcScore,
    voice_id: str,
    measure: Measure,
    unit_denominator: int,
) -> str:
    voice = score.voice_arrs[voice_id]
    show_chords = voice_id == "Vocal"
    measure_accidentals = {}
    current_key = str(score.key_arr[measure.start_t])
    key_accidentals = get_key_accidentals(current_key)
    parts = []
    padding = _measure_padding_units(measure, unit_denominator)
    if padding < 0:
        raise AbcRebuildError(
            f"Measure {measure.index}: notated meter is shorter than its decoded span"
        )
    leading_padding = padding if measure.pad_before else 0
    trailing_padding = 0 if measure.pad_before else padding
    t = measure.start_t
    while t < measure.end_t:
        change_points = [measure.end_t]
        for probe in range(t + 1, measure.end_t):
            if not _same_note_segment(int(voice[t]), int(voice[probe])):
                change_points.append(probe)
                break
        for probe in range(t + 1, measure.end_t):
            if score.key_arr[probe] != score.key_arr[probe - 1]:
                change_points.append(probe)
                break
        if show_chords:
            for probe in range(t + 1, measure.end_t):
                if score.chord_arr[probe] != score.chord_arr[probe - 1]:
                    change_points.append(probe)
                    break
        next_t = min(change_points)

        prefix = ""
        key = str(score.key_arr[t])
        if t > measure.start_t and key != current_key:
            current_key = key
            key_accidentals = get_key_accidentals(current_key)
            measure_accidentals = {}
            prefix += f"[K:{current_key}]"

        if show_chords and (t == measure.start_t or score.chord_arr[t] != score.chord_arr[t - 1]):
            chord = str(score.chord_arr[t])
            chord_text = chord_symbol_to_abc(chord)
            if chord_text is not None:
                prefix += f'"{chord_text}"'

        value = int(voice[t])
        if value == 0:
            note_text = "z"
        else:
            note_text = note_to_abc(value // 2 - 1, key_accidentals, measure_accidentals)
        duration = _duration_units(score, t, next_t, unit_denominator)
        if t == measure.start_t and leading_padding:
            if value == 0 and not prefix:
                duration += leading_padding
            else:
                parts.extend(
                    _render_duration_tokens(
                        "",
                        "z",
                        leading_padding,
                        tie_out=False,
                    )
                )
            leading_padding = 0
        if value == 0 and next_t == measure.end_t and trailing_padding:
            duration += trailing_padding
            trailing_padding = 0
        if duration <= 0:
            raise AbcRebuildError(f"Non-positive ABC duration at subbeats {t}:{next_t}")
        tie_out = (
            value > 0
            and next_t < len(voice)
            and _continues_pitch(value, int(voice[next_t]))
        )
        parts.extend(
            _render_duration_tokens(
                prefix,
                note_text,
                duration,
                tie_out=tie_out,
            )
        )
        t = next_t
    if leading_padding:
        raise AbcRebuildError(
            f"Measure {measure.index}: leading rest padding was not serialized"
        )
    if trailing_padding:
        parts.extend(
            _render_duration_tokens(
                "",
                "z",
                trailing_padding,
                tie_out=False,
            )
        )
    return "".join(parts)

def _is_compressible_full_rest(rendered_measure: str) -> bool:
    """Whether a rendered measure can be losslessly replaced by ABC ``Z``."""

    cursor = 0
    saw_note = False
    for match in _MUSIC_ELEMENT_RE.finditer(rendered_measure):
        if rendered_measure[cursor:match.start()]:
            return False
        cursor = match.end()
        if match.group("quoted") is not None or match.group("key") is not None:
            return False
        saw_note = True
        if match.group("note") != "z" or match.group("tie"):
            return False
    return saw_note and cursor == len(rendered_measure)

def _render_voice_group(
    score: RebuiltAbcScore,
    voice_id: str,
    measures: list[Measure],
    unit_denominator: int,
) -> str:
    rendered = [
        _render_voice_measure(
            score,
            voice_id,
            measure,
            unit_denominator,
        )
        for measure in measures
    ]
    parts = []
    index = 0
    while index < len(rendered):
        if not _is_compressible_full_rest(rendered[index]):
            parts.append(rendered[index] + "|")
            index += 1
            continue
        end = index + 1
        while (
            end < len(rendered)
            and _is_compressible_full_rest(rendered[end])
        ):
            end += 1
        count = end - index
        parts.append("Z" + (str(count) if count > 1 else "") + "|")
        index = end
    return "".join(parts)

def _sanitize_structure_label(value: str) -> str:
    return " ".join(str(value).split())

def _measure_groups(score: RebuiltAbcScore) -> list[MeasureGroup]:
    first_measure = score.measures[0]
    active_meter = (
        first_measure.abc_numerator,
        first_measure.abc_denominator,
    )
    active_key = str(score.key_arr[first_measure.start_t])
    active_structure = ""
    groups: list[MeasureGroup] = []

    for measure in score.measures:
        meter = (measure.abc_numerator, measure.abc_denominator)
        key = str(score.key_arr[measure.start_t])
        meter_changed = meter != active_meter
        key_changed = key != active_key
        new_structure_labels = []
        for t, label in score.structure_events:
            if not measure.start_t <= t < measure.end_t:
                continue
            clean_label = _sanitize_structure_label(label)
            if clean_label and clean_label != active_structure:
                new_structure_labels.append(clean_label)
                active_structure = clean_label

        start_group = (
            not groups
            or len(groups[-1].measures) >= 4
            or meter_changed
            or key_changed
            or bool(new_structure_labels)
        )
        if start_group:
            groups.append(
                MeasureGroup(
                    measures=[measure],
                    structure_labels=new_structure_labels,
                    meter_changed=meter_changed,
                    key_changed=key_changed,
                )
            )
        else:
            groups[-1].measures.append(measure)

        active_meter = meter
        active_key = str(score.key_arr[measure.end_t - 1])
    return groups

def score_to_abc(score: RebuiltAbcScore) -> str:
    unit_denominator = abc_unit_denominator(score)
    first_measure = score.measures[0]
    first_key = str(score.key_arr[first_measure.start_t])
    lines = [
        "X:1",
        "T:",
        f"M:{first_measure.abc_numerator}/{first_measure.abc_denominator}",
        f"L:1/{unit_denominator}",
        f"Q:1/4={int(round(estimate_tempo(score)))}",
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
        'V: Ins clef=treble name="Ins Melody" snm="Inst."',
        f"K:{first_key}",
    ]
    for group in _measure_groups(score):
        lines.extend(f"% {label}" for label in group.structure_labels)
        first_group_measure = group.measures[0]
        for voice_id in VOICE_IDS:
            lines.append(f"V: {voice_id}")
            if group.meter_changed:
                lines.append(
                    f"M:{first_group_measure.abc_numerator}/"
                    f"{first_group_measure.abc_denominator}"
                )
            if group.key_changed:
                lines.append(
                    f"K:{score.key_arr[first_group_measure.start_t]}"
                )
            lines.append(
                _render_voice_group(
                    score,
                    voice_id,
                    group.measures,
                    unit_denominator,
                )
            )
    text = "\n".join(lines) + "\n"
    return text
