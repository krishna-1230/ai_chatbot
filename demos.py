from flask import Flask, render_template, request
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD

# Load intents from JSON file
with open("intents.json") as f:
    intent_data = json.load(f)["intents"]
    print(len(intent_data[0]))
# Preprocess the data
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_chars = ["?", "!", ".", ","]

for intents in intent_data:
    for intent, intent_values in intents.items():
        for pattern in intent_values["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent))
            if intent not in classes:
                classes.append(intent)

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training_data = []
output_data = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training_data.append(bag)
    output_data.append(output_row)

training_data = np.array(training_data)
output_data = np.array(output_data)

# Build the model
model = Sequential()
model.add(Dense(32768, input_shape=(len(training_data[0]),), activation='relu'))
print(len(training_data[0]))
model.add(Dropout(0.5))
model.add(Dense(8192, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8192, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output_data[0]), activation='softmax'))
print(len(output_data[0]))
adam = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

sgd = SGD(learning_rate=0.6, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(training_data, output_data, epochs=2000, batch_size=600, verbose=1)



#
# model = Sequential()
# model.add(Dense(8192, input_shape=(len(training_data[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(8192, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(8192, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(output_data[0]), activation='softmax'))
#
# adam = Adam(learning_rate=0.0001)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#
# model.fit(training_data, output_data, epochs=2000, batch_size=1200, verbose=1)