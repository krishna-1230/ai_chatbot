# import nltk
# nltk.download('punkt')
# nltk.download('wordnet') 
# nltk.download('punkt_tab')

import nltk
import tensorflow as tf

# Download required NLTK data packages
nltk.download('punkt')
nltk.download('wordnet')

# Check GPU availability for TensorFlow
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

print("NLTK data packages successfully downloaded.")