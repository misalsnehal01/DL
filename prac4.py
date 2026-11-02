import tensorflow as tf
import numpy as np

# 1. Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 2. Define Autoencoder
input_layer = tf.keras.layers.Input(shape=(784,))

# 3. Encoder
encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)

# 4. Decoder
decoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(decoded)

# Create model
autoencoder = tf.keras.Model(input_layer, decoded)

# 5. Compile model
autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train model
autoencoder.fit(
    x_train, x_train,
    epochs=5,
    batch_size=256,
    validation_data=(x_test, x_test)
)

# Reconstruct test images
predictions = autoencoder.predict(x_test)

# Calculate reconstruction error
error = np.mean(np.square(x_test - predictions), axis=1)

# Detect anomalies
threshold = np.percentile(error, 95)

anomalies = error > threshold

print("Total Anomalies:", np.sum(anomalies))
print("Normal Samples:", np.sum(~anomalies))