import tensorflow as tf
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess data
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Data Loaded Successfully")
model = tf.keras.Sequential([
    tf.keras.Input(shape=(32,32,3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model Created Successfully")
history = model.fit(x_train, y_train, epochs=5)

print("Model Training Completed")
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'],label='Loss')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy/Loss")
plt.show()