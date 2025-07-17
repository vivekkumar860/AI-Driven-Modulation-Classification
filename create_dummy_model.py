import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a simple sequential model
model = keras.Sequential([
    layers.Input(shape=(128, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(11, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Save the model
model.save('mod_classifier.h5')

print('Dummy model saved as mod_classifier.h5') 