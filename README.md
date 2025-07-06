# AI Chatbot with Web Interface

An intelligent chatbot application built with Flask and TensorFlow, featuring natural language processing, user authentication, and a responsive web interface.

![Chatbot Interface](static/builtin/chat.png)

## Features

- **Neural Network-Based Conversations**: Deep learning model trained on custom intents for natural responses
- **User Authentication System**: Secure login and registration with data persistence
- **OpenAI Integration**: Enhanced responses using OpenAI's API (optional)
- **Conversation History**: Persistent chat history across sessions
- **Contact Form**: Integrated email functionality for user inquiries
- **Responsive Design**: Modern web interface that works across all devices

## Tech Stack

- **Backend**: Python 3.10+, Flask 3.1+
- **AI/ML**: TensorFlow 2.10+, Keras, NLTK 3.9+
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Storage**: JSON for intent patterns and user information
- **Email Integration**: SMTP for contact form functionality

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)
- Gmail account (for contact form functionality)

## Installation and Setup

1. **Clone the repository** (or download and extract the ZIP file)
   ```bash
   git clone https://github.com/krishna-1230/ai_chatbot.git
   cd chatbot_ai
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLTK data**
   ```bash
   python nltk_download.py
   ```

5. **Set up environment variables**
   
   Create a `.env` file in the project root with the following variables:
   ```
   GMAIL_PASS=your_email_password
   OPENAI_KEY=your_openai_api_key  # Optional, for enhanced responses
   ```

   Alternatively, set them directly in your environment:
   ```bash
   # Windows
   set GMAIL_PASS=your_email_password
   set OPENAI_KEY=your_openai_api_key

   # macOS/Linux
   export GMAIL_PASS=your_email_password
   export OPENAI_KEY=your_openai_api_key
   ```

6. **Run the application**
   ```bash
   # Windows (using the batch file)
   run_chatbot.bat
   
   # Or directly with Python
   python main.py
   ```

7. **Access the application** in your web browser at `http://localhost:5000`

## Project Structure

```
chatbot_ai/
├── data.json                # User authentication data
├── demos.py                 # Demonstration scripts
├── history_chat.json        # Chat history storage
├── intents.json             # Training data for the chatbot
├── K.py                     # Alternative implementation with OpenAI
├── main.py                  # Main Flask application
├── nltk_download.py         # Script to download NLTK data
├── requirements.txt         # Project dependencies
├── run_chatbot.bat          # Windows launcher script
├── static/                  # Static assets
│   ├── builtin/            # Built-in images
│   └── uploads/            # User uploads
└── templates/               # HTML templates
    ├── about.html
    ├── base.html
    ├── chatbot.html
    └── ...
```

## Usage

1. **Registration/Login**: Create an account or sign in with existing credentials
2. **Chat Interface**: Interact with the AI chatbot through the user-friendly interface
3. **View History**: Access your past conversations in the history section
4. **Contact**: Reach out through the contact form for support or inquiries

## Customization

### Training the Model with Custom Intents

1. Edit the `intents.json` file to add new patterns and responses
2. Run the training script to update the model:
   ```bash
   python demos.py
   ```

### Switching Between Models

The project includes two main implementations:
- `main.py`: Standard neural network model
- `K.py`: Enhanced model with OpenAI integration

To use the OpenAI-enhanced version:
1. Ensure you have set the `OPENAI_KEY` environment variable
2. Run `python K.py` instead of `main.py`

## Security Considerations

- User passwords are stored in plain text in `data.json` - in a production environment, implement proper password hashing
- For production use, implement proper session management and CSRF protection
- Consider using a database instead of JSON files for data storage in production

## Future Enhancements

- Database integration (SQLite, PostgreSQL)
- Password hashing and enhanced security
- Multi-language support
- Voice interaction capabilities
- Personalized responses based on user history
- Docker containerization

## Troubleshooting

- **NLTK Data Issues**: If you encounter NLTK-related errors, run `python nltk_download.py`
- **Email Sending Failures**: Ensure your Gmail account allows less secure apps or use App Passwords
- **Model Training Issues**: Try reducing batch size in `demos.py` if you encounter memory errors

## License

[MIT License](LICENSE)

## Contact

For questions, support, or contributions, please:
- Submit an issue on GitHub
- Contact through the application's contact form
- Email: your.email@example.com (replace with your actual contact email) 