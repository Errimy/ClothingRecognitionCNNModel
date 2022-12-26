import tensorflow as tf

# Load the TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model("HatimCNNModelfotTFL.py")

# Convert the model
tflite_model = converter.convert()

# Save the model to a file
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)