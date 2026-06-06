from music21 import converter, note, chord
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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
n_vocab = len(pitchnames)

note_to_int = dict(
    (note, number)
    for number, note in enumerate(pitchnames)
)

sequence_length = 25

network_input = []
network_output = []

for i in range(len(all_notes) - sequence_length):

    seq_in = all_notes[i:i + sequence_length]
    seq_out = all_notes[i + sequence_length]

    network_input.append(
        [note_to_int[n] for n in seq_in]
    )

    network_output.append(
        note_to_int[seq_out]
    )

n_patterns = len(network_input)

network_input = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

network_input = network_input / float(n_vocab)

network_output = to_categorical(network_output)

model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(network_input.shape[1],
                     network_input.shape[2])
    )
)

model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dense(n_vocab, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="loss",
    save_best_only=True
)
history = model.fit(
    network_input,
    network_output,
    epochs=5,
    batch_size=64,
    callbacks=[checkpoint]
)
model.save("music_model.keras")

print("Training completed!")