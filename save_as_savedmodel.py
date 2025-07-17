import tensorflow as tf

model = tf.keras.models.load_model("mod_classifier.h5")
model.export("saved_model")  # Use export for SavedModel in Keras 3.x
print("Model exported as TensorFlow SavedModel in 'saved_model/'") 