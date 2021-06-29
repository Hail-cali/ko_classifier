import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
import os
import sys

timesteps = 10
input_dim = 4
hidden_size = 8

inputs = np.random.random((timesteps, input_dim))

print(inputs)

hidden_state_t = np.zeros((hidden_size,))

Wx = np.random.random((hidden_size, input_dim))
Wh = np.random.random((hidden_size, hidden_size))
b = np.random.random((hidden_size,))

total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t))
    print(np.shape(total_hidden_states))
    hidden_state_t = output_t
total_hidden_states = np.stack(total_hidden_states, axis=0)

print(total_hidden_states)

model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8,2,10), return_sequences=True))
print(model.summary())
