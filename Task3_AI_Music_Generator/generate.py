from music21 import converter, note, chord
from tensorflow.keras.models import load_model
import numpy as np
import os
import random

# Read all notes again
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

int_to_note = dict(
    (number, note)
    for number, note in enumerate(pitchnames)
)

sequence_length = 50

network_input = []

for i in range(len(all_notes) - sequence_length):
    sequence = all_notes[i:i + sequence_length]
    network_input.append([note_to_int[n] for n in sequence])

n_vocab = len(pitchnames)

# Load model
model = load_model("best_model.keras")

# Pick random starting point
start = random.randint(0, len(network_input)-1)

pattern = network_input[start]

prediction_output = []

# Generate 200 notes
for note_index in range(200):

    prediction_input = np.reshape(
        pattern,
        (1, len(pattern), 1)
    )

    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(
        prediction_input,
        verbose=0
    )

    prediction = prediction[0]

index = np.random.choice(
    len(prediction),
    p=prediction / np.sum(prediction)
)

result = int_to_note[index]

prediction_output.append(result)

pattern.append(index)
pattern = pattern[1:]

print("Generated notes:")
print(prediction_output[:20])

from music21 import stream, note, chord

offset = 0
output_notes = []

for pattern in prediction_output:

    if '.' in pattern:
        notes_in_chord = pattern.split('.')
        chord_notes = []

        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.offset = offset
            chord_notes.append(new_note)

        new_chord = chord.Chord(chord_notes)
        output_notes.append(new_chord)

    else:
        try:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)
        except:
            pass

    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music.mid')

print("MIDI file saved!")