# AI Chatbot with Web Interface

An intelligent chatbot application built with Flask, featuring user authentication, intent recognition using deep learning, and a responsive web interface.

## Features

- **Intelligent Conversations**: Neural network model trained to understand user intents and provide relevant responses
- **User Authentication**: Secure login and signup functionality
- **Conversation History**: Save and view past conversations
- **Contact Form**: Integrated email functionality for user inquiries
- **Responsive Design**: Modern web interface that works across devices

## Tech Stack

- **Backend**: Python, Flask
- **AI/ML**: Keras, NLTK, Neural Networks
- **Frontend**: HTML, CSS, JavaScript
- **Data**: JSON for intent patterns and user information

## Installation and Setup

1. Clone the repository
   ```
   git clone https://github.com/yourusername/chatbot_ai.git
   cd chatbot_ai
   ```

2. Install dependencies
   ```
   pip install flask numpy nltk keras tensorflow
   ```

3. Download NLTK data
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

4. Set up email functionality (for contact form)
   ```
   # Create environment variable for your email password
   # For Windows:
   set GMAIL_PASS=your_email_password
   
   # For Linux/Mac:
   export GMAIL_PASS=your_email_password
   ```

5. Run the application
   ```
   python main.py
   ```

6. Access the application in your browser at `http://localhost:5000`

## Usage

1. **Login/Signup**: Create an account or log in with existing credentials
2. **Chat Interface**: Interact with the AI chatbot through a user-friendly interface
3. **View History**: Access your past conversations
4. **Contact**: Reach out through the contact form for any inquiries

## Project Structure

- `main.py` - Main application file containing Flask routes and ML model
- `intents.json` - Training data for the chatbot
- `data.json` - User authentication data
- `templates/` - HTML templates for the web interface
- `static/` - Static files (CSS, JavaScript, images)

## Future Enhancements

- Enhanced NLP capabilities with more training data
- Multi-language support
- Voice interaction
- More personalized responses based on user history

## License

[MIT License](LICENSE)

## Contact

For questions and support, please submit an issue or contact through the application's contact form. 