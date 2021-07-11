from train import *
import streamlit as st
import pickle
from PIL import Image
from tensorflow import keras

import re
image = Image.open('img.webp')
simple_rnn_model = keras.models.load_model('model.h5')

def final_predictions(text):
    y_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
    y_id_to_word[0] = '<PAD>'
    sentence = [english_tokenizer.word_index[word] for word in text.split()]
    sentence = pad_sequences([sentence], maxlen=preproc_french_sentences.shape[-2], padding='post')
  
#   print(sentence.shape)
    return (logits_to_text(simple_rnn_model.predict(sentence[:1])[0], french_tokenizer))

def main():
    st.title("English to French translator")
    st.write(" ")
    st.write(" ")
    st.image(image)
    st.write(" ")
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #5F939A
    }
    </style>
    """, unsafe_allow_html=True)
    
    inputs = st.text_input("Enter the text in english")
    st.write(" ")
    if st.button('Transate'):
        txt=inputs.lower()
        x = final_predictions(re.sub(r'[^\w]', ' ', txt))
        x = x.split(" ")
        y = []
        for i in range(len(x)):
            if x[i]!='<PAD>':
                y.append(x[i])
        x = " ".join(y)
        st.error(f'''Your text in french is:
                        {x}.''')

if __name__=='__main__':
    main()