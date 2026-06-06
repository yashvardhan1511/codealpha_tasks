from music21 import converter, note, chord
import os

all_notes = []

for file in os.listdir("dataset"):
    if file.endswith(".mid") or file.endswith(".midi"):

        print("Reading:", file)

        midi = converter.parse("dataset/" + file)

        for n in midi.flatten().notes:

            if isinstance(n, note.Note):
                all_notes.append(str(n.pitch))

            elif isinstance(n, chord.Chord):
                all_notes.append('.'.join(str(p) for p in n.normalOrder))

print("\nTotal notes found:", len(all_notes))

print("\nFirst 20 notes:")
print(all_notes[:20])