import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


train_data = []
train_labels = []

test_data = []
test_labels = []

def load_data(name, label):
  saltue = open('data/' + name + '.dat', 'rb')

  for i in range(0, 250):
    train_data.append(np.load(saltue).tolist())
    train_labels.append(label)

  for i in range (250, 300):
    test_data.append(np.load(saltue).tolist())
    test_labels.append(label)


model = Sequential()
model.add(Flatten())
model.add(Dense(20, input_dim = 42, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))

  


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


load_data('fist', 0)
load_data('stop', 1)
load_data('peace', 2)
load_data('thumbsup', 3)

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

model.fit(train_data, train_labels, epochs=10, shuffle=True, batch_size=10)

model.save('./gesture_model_mac')

model.evaluate(test_data, test_labels, verbose=1)
