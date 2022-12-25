# TensorFlow and tf.keras

import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from keras import models


print(tf.__version__)

# Step 1: Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)

# Step 2: Create the Model
model = models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', input_shape=(28, 28,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
#Add Dense layers on top
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10)


# Step 3: Train the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_data = tf.squeeze(train_images)
test_data = tf.squeeze(test_images)
model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

#Evaluate the model

# Step 4: Test the Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)





# Save the Model
model.save(r'C:\Users\redat\PycharmProjects\CNNFashionMnist\CNNLab\Deploy_Model\static\model\HatimCNNModel.h5')
print('Model saved successfully !')



