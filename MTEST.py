import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
import matplotlib.pyplot as plt

import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from collections import Counter

# Ensure required NLTK packages are downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
with open("intents.json") as f:
    intent_data = json.load(f)
    intents = intent_data["intents"]

# Preprocess the data
words = []
classes = []
documents = []
ignore_chars = ["?", "!", ".", ","]

# Extract patterns and intents
for intent_group in intents:
    for intent in intent_group:
        for pattern in intent_group[intent]["patterns"]:
            # Tokenize each pattern
            tokens = word_tokenize(pattern)
            # Add tokens to words list
            words.extend(tokens)
            # Add documents to corpus
            documents.append((tokens, intent))
            # Add intent to classes if not already there
            if intent not in classes:
                classes.append(intent)

# Lemmatize, lowercase, and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"Unique lemmatized words: {len(words)}")
print(f"Intent classes: {len(classes)}")
print(f"Training documents: {len(documents)}")

# Count examples per class
class_counts = Counter([doc[1] for doc in documents])
single_example_classes = [cls for cls, count in class_counts.items() if count == 1]
print(f"Classes with only one example: {len(single_example_classes)}")

# Create training data
training_data = []
output_data = []
output_empty = [0] * len(classes)

# Create bag of words for each document
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training_data.append(bag)
    output_data.append(output_row)

# Convert to numpy arrays
X = np.array(training_data)
y = np.array(output_data)

# Split the data into training and validation sets
# Use simple random split instead of stratified split due to single-example classes
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")


# Define model building function with parameters suitable for high-dimensional, sparse data
def build_model(input_shape, output_shape, dropout_rate=0.3, l2_reg=0.0001):
    model = Sequential()

    # Input layer - larger for sparse data
    model.add(Dense(1024, input_shape=(input_shape,), activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers
    model.add(Dense(2048, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Add another layer for more capacity with many classes
    model.add(Dense(2048, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Final hidden layer
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer - one neuron per class
    model.add(Dense(output_shape, activation='softmax'))

    # Compile model with Adam optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for stability
        metrics=['accuracy']
    )

    return model


# Build the model
model = build_model(len(X_train[0]), len(classes))
print(model.summary())

# Define callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,  # Increased patience for many classes
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'intent_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train the model with early stopping and learning rate reduction
history = model.fit(
    X_train, y_train,
    epochs=300,  # More epochs with early stopping
    batch_size=32,  # Smaller batch size for better generalization with many classes
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Save the model, words, and classes for later use
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

model.save('intent_model.h5')

print("Model, words, and classes saved to disk")


# Function to predict intent from text
def predict_intent(text, model, words, classes):
    # Clean up the text
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)]

    # Create bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    # Get prediction
    results = model.predict(np.array([bag]))[0]

    # Filter out predictions below threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


# Test function
if len(documents) > 0:
    # Get a sample pattern for testing
    sample_idx = 0
    test_intent = list(intent_data["intents"][0].keys())[0]
    test_text = intent_data["intents"][0][test_intent]["patterns"][0]
    prediction = predict_intent(test_text, model, words, classes)
    print(f"Test text: {test_text}")
    print(f"Prediction: {prediction}")