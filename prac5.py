import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. Prepare data
sentences = [
    "I love machine learning",
    "I love deep learning",
    "Machine learning is interesting"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# 2. Generate training data
sequences = tokenizer.texts_to_sequences(sentences)

data = []

for seq in sequences:
    for i in range(1, len(seq)-1):
        context = [seq[i-1], seq[i+1]]
        target = seq[i]
        data.append((context, target))

X = np.array([x[0] for x in data])
y = np.array([x[1] for x in data])

# 3. Build CBOW model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=10),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(vocab_size, activation='softmax')
])

# 4. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train model
model.fit(X, y, epochs=100, verbose=0)

# 6. Output
print("Vocabulary:", word_index)
print("Training Data:", X)
print("Target Words:", y)

print("CBOW Model Training Completed")