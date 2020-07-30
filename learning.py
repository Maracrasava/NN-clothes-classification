import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                        'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

# print(len(test_labels)): 10000 images in training set

# Scaling data
train_images = train_images / 255
test_images = test_images / 255

# Building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compiling the model
model.compile(optimizer='adam',  # Stochastic Gradient decent
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluating accuracy using test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('Test accuracy:', test_acc)

# Making predictions
probability_model = keras.Sequential([model, tf.keras.layers.Softmax()])
# Here, the model has predicted the label for each image in the testing set.
predictions = probability_model.predict(test_images)
print(f"Prediction: {class_names[np.argmax(predictions[0])]}, Actual: {class_names[test_labels[0]]}")



