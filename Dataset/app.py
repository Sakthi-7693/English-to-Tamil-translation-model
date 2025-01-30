import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load your pre-trained model (encoder and decoder)
model = load_model('eng_to_tamil_translation_model.h5')  # Update with your model file path

# Initialize the tokenizers for English and Tamil (use the same tokenizers from your training phase)
english_tokenizer = Tokenizer()
tamil_tokenizer = Tokenizer()

# Assuming these are the maximum lengths from your training
max_english_length = 100  # You can replace this with the actual max length from your training phase
max_tamil_length = 100    # Replace with the actual max length

# Function to translate an English sentence to Tamil
def translate_sentence(input_text):
    input_seq = english_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_english_length, padding='post')

    states_value = model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tamil_tokenizer.word_index.get('<start>', 1)

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tamil_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence) > max_tamil_length:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return " ".join(decoded_sentence).strip()

# Streamlit Web Application Layout
st.title('English to Tamil Translator')
st.write('Enter an English sentence to translate it into Tamil')

# Create two columns for input and output
col1, col2 = st.columns(2)

with col1:
    # Input text box for English sentence
    english_input = st.text_area("Enter English Sentence", "")

with col2:
    # When the user clicks the translate button
    if st.button("Translate"):
        if english_input:
            tamil_output = translate_sentence(english_input)
            st.write("### Translated Tamil Sentence")
            st.write(tamil_output)
        else:
            st.write("Please enter an English sentence to translate.")
