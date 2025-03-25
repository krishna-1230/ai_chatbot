from flask import Flask, render_template, request
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import os

password = os.environ.get('GMAIL_PASS')


def send_mail_function(w, x, y, z):
    """w,x,y,z are the name, email, phone, and message of the customer. this function sends mail to my gmail acc"""
    message = MIMEMultipart()
    message['From'] = 'project.dt.krish@gmail.com'
    message['To'] = 'project.dt.krish@gmail.com'
    message['Subject'] = 'Test email from Python'
    body = f'Name: {w} \nEmail: {x} \nPhone no: {y} \nMessage: {z}'
    message.attach(MIMEText(body, 'plain'))

    # Connect to Gmail's SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    server.login('project.dt.krish@gmail.com', password)

    # Send the email
    text = message.as_string()
    server.sendmail('project.dt.krish@gmail.com', 'project.dt.krish@gmail.com', text)

    # Close the SMTP server connection
    server.quit()


# Read the intents from a JSON file
with open("intents.json") as f:
    intentt = json.load(f)
    intentss = intentt["intents"]

# Preprocess the data
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_chars = ["?", "!", ".", ","]

for intents in intentss:
    for intent in intents:
        for pattern in intents[intent]["patterns"]:
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
#====================================================================
# Build the model
model = Sequential()
model.add(Dense(1024, input_shape=(len(training_data[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))

# Adding another layer
model.add(Dense(2048, activation='relu'))  # You can adjust the number of neurons as needed
model.add(Dropout(0.5))  # Adding dropout for regularization
model.add(Dense(1024, activation='relu'))  # You can adjust the number of neurons as needed
model.add(Dropout(0.5))



model.add(Dense(len(output_data[0]), activation='softmax'))

sgd = SGD(learning_rate=0.6, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(training_data, output_data, epochs=500, batch_size=5000, verbose=1)




#=====================================================================================================================

#=====================================================================================================================

# Define the function to predict the intent
def predict_intent(text):
    bag = []
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    for word in words:
        bag.append(1) if word in tokens else bag.append(0)
    input_data = np.array([bag])
    result = model.predict(input_data)[0]
    threshold = 0.25
    if np.max(result) > threshold:
        output = classes[np.argmax(result)]
    else:
        output = "unknown"
    return output


app = Flask(__name__)


@app.route('/')
def first():
    """ this is the first page of the website """
    return render_template('first.html')


@app.route('/getstarted')
def second():
    """ this is the page which shows login or signup """
    return render_template("second.html")


@app.route('/login')
def third_a():
    """ if login button is clicked, it will show the login page """
    return render_template("login.html")


@app.route('/login_name_pass', methods=['POST'])  # login button goes here to check user and password
def login_a_b():
    """ this will get the email and password from the respective fields and checks the json file called data.json
    which has the username and password when signed in from sign-in button"""
    if request.method == 'POST':
        n = request.form.get('username')
        a = request.form.get('password')
        e = request.form.get('customerEmail')
        with open("data.json") as f:
            data = json.load(f)
            for user in data['intents']:
                if user['username'] == str(n) and user['password'] == str(a) and user['email'] == str(e):
                    return render_template(
                        'main1.html')  # if user exists after checking the data.json file it goes to this file

            return render_template(
                'login_a.html')  # if doesn't exist it goes to a page "login_a.html" it displays user not found and
            # will display the same page until the user isw found


@app.route('/signup')
def signup():
    """ this is the sign up page . After entering the username and password, press sign up. this will go to the below
     function """
    return render_template("signup.html")


@app.route('/signup_name_pass', methods=['post'])
def signup_getting():
    """if sign up button is pressed it will go to this page. the function will check if the username already exists in
    the data.json file if the user exists it will go to "signup_a.html" . if the name is available then it will append
    to the "data.json" file and displays the main1.html """
    if request.method == 'POST':
        n = request.form.get('username')
        a = request.form.get('password')
        e = request.form.get('customerEmail')
        with open('data.json', 'r') as f:
            is_there = False
            data = json.load(f)
            print(data)
            for user in data['intents']:
                if user['username'] == str(n):
                    is_there = True
                    return render_template('signup_a.html')  # if name exists
            if not is_there:
                new_data = {"username": str(n), "password": str(a), "email": str(e)}
                data["intents"].append(new_data)
                with open('data.json', "w") as f:
                    json.dump(data, f)
                    return render_template('main1.html')  # if name is available


@app.route('/home')
def home1():
    return render_template("main1.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/thankyou', methods=['POST'])
def contact_post():
    if request.method == 'POST':
        n = request.form.get('customerName')
        e = request.form.get('customerEmail')
        m = request.form.get('customerPhone')
        mess = request.form.get('customerNote')
        send_mail_function(n, e, m, mess)

        return render_template("thankyou_contact.html")  # , name=n, email=e, mobile=m, message=mess


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        form_data = request.form
        json_dump = json.dumps(form_data, indent=4)
        json_object = json.loads(json_dump)
        text = json_object["user_input"]
        intent = predict_intent(text)
        if intent == "unknown":
            with open("response.json", 'r+') as file:
                file_data = json.load(file)
                dictionary = {
                    "you": text,
                    "response": "I'm sorry, I don't understand."
                }
                file_data["bot"].append(dictionary)
                file.seek(0)
                json.dump(file_data, file, indent=4)
        else:
            for intents in intentss:
                response = np.random.choice(intents[intent]["responses"])
                print(response)
                with open("response.json", 'r+') as file:
                    file_data = json.load(file)
                    dictionary = {
                        "you": text,
                        "response": response
                    }
                    file_data["bot"].append(dictionary)
                    file.seek(0)
                    json.dump(file_data, file, indent=4)

    with open("response.json") as f:
        data_query = json.load(f)
        chatlist = data_query["bot"]
        return render_template('chatbot.html', chatlist=chatlist)


@app.route('/chat_clear', methods=['GET', 'POST'])
def chatclear():
    with open('response.json', 'r') as source_file:
        source_data = json.load(source_file)

    with open('history_chat.json', 'r+') as dest_file:
        dest_data = json.load(dest_file)
        dest_data["bot"].extend(source_data["bot"])
        dest_file.seek(0)
        json.dump(dest_data, dest_file)
        dest_file.truncate()
    with open("response.json", "w") as g:
        data = {"bot": []}
        json.dump(data, g)
    with open("response.json") as f:
        data_query = json.load(f)
        chatlist = data_query["bot"]
    return render_template('chatbot.html', chatlist=chatlist)


@app.route('/chat_hist', methods=['GET', 'POST'])
def chat_hist():
    with open("history_chat.json") as f:
        data_query = json.load(f)
        chatlist = data_query["bot"]
        return render_template('hist.html', chatlist=chatlist)


@app.route('/hist_clear', methods=['GET', 'POST'])
def hist_clear():
    with open("history_chat.json", "w") as g:
        data = {"bot": []}
        json.dump(data, g)
    with open("history_chat.json") as f:
        data_query = json.load(f)
        chatlist = data_query["bot"]
    return render_template('hist.html', chatlist=chatlist)




app.run(debug=True)
