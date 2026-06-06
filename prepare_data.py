from music21 import converter, note, chord
import os
import numpy as np

all_notes = []

for file in os.listdir("dataset"):
    if file.endswith(".mid") or file.endswith(".midi"):

        midi = converter.parse("dataset/" + file)

        for n in midi.flatten().notes:

            if isinstance(n, note.Note):
                all_notes.append(str(n.pitch))

            elif isinstance(n, chord.Chord):
                all_notes.append('.'.join(str(p) for p in n.normalOrder))

pitchnames = sorted(set(all_notes))

note_to_int = dict(
    (note, number)
    for number, note in enumerate(pitchnames)
)

sequence_length = 50

network_input = []
network_output = []

for i in range(len(all_notes) - sequence_length):

    seq_in = all_notes[i:i + sequence_length]

    seq_out = all_notes[i + sequence_length]

    network_input.append(
        [note_to_int[note] for note in seq_in]
    )

    network_output.append(
        note_to_int[seq_out]
    )

print("Training patterns:", len(network_input))
print("Example input:")
print(network_input[0][:10])

print("Expected output:")
print(network_output[0])