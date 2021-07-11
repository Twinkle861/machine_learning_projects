#Now importing modules
# import helper
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy


# Load the dataset and split file by lines
import os
def load_data(path):
  input_file = os.path.join(path)
  with open(input_file, "r") as f:
    data = f.read()
  return data.split('\n')

english_sentences=load_data("data/small_vocab_en.csv")
french_sentences=load_data("data/small_vocab_fr.csv")
# print("done")
# for i in range(5):
#   print('Sample :',i)
#   print(english_sentences[i])
#   print(french_sentences[i])
#   print('-'*50)


# The complexity of the problem is determined by the complexity of the vocabulary. A more complex vocabulary is a more complex problem
import collections
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])
# print('English Vocab:',len(english_words_counter))
# print('French Vocab:',len(french_words_counter))


# For a neural network to predict on text data, it first has to be turned into data it can understand. Text data like "dog" is a sequence of ASCII character encodings. Since a neural network is a series of multiplication and addition operations, the input data needs to be number(s).We can turn each character into a number or each word into a number. These are called character and word ids, respectively. Character ids are used for character level models that generate text predictions for each character. A word level model uses word ids that generate text predictions for each word. Word level models tend to learn better, since they are lower in complexity, so we'll use those.Turn each sentence into a sequence of words ids using Keras's Tokenizer function. Use this function to tokenize english_sentences and french_sentences in the cell below.
def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

# # Tokenize Sample output
# text_sentences = [
#     'The quick brown fox jumps over the lazy dog .',
#     'By Jove , my quick study of lexicography won a prize .',
#     'This is a short sentence .']
# text_tokenized, text_tokenizer = tokenize(text_sentences)
# print(text_tokenizer.word_index)
# print()
# for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
#   print('Sequence {} in x'.format(sample_i + 1))
#   print('  Input:  {}'.format(sent))
#   print('  Output: {}'.format(token_sent))


# When batching the sequence of word ids together, each sequence needs to be the same length. Since sentences are dynamic in length, we can add padding to the end of the sequences to make them the same length.Make sure all the English sequences have the same length and all the French sequences have the same length by adding padding to the end of each sequence using Keras's pad_sequences function.
def pad(x, length=None):
  return pad_sequences(x, maxlen=length, padding='post')


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    #Expanding dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)
# print('Data Preprocessed')
# print("Max English sentence length:", max_english_sequence_length)
# print("Max French sentence length:", max_french_sequence_length)
# print("English vocabulary size:", english_vocab_size)
# print("French vocabulary size:", french_vocab_size)


# The neural network will be translating the input to words ids, which isn't the final form we want. We want the French translation. The function logits_to_text will bridge the gab between the logits from the neural network to the French translation
def logits_to_text(logits, tokenizer):
  index_to_words = {id: word for word, id in tokenizer.word_index.items()}
  index_to_words[0] = '<PAD>'

  #So basically we are predicting output for a given word and then selecting best answer
  #Then selecting that label we reverse-enumerate the word from id
  return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


if __name__=='__main__':
    
  def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
      """
      Build and train a RNN model using word embedding on x and y
      :param input_shape: Tuple of input shape
      :param output_sequence_length: Length of output sequence
      :param english_vocab_size: Number of unique English words in the dataset
      :param french_vocab_size: Number of unique French words in the dataset
      :return: Keras model built, but not trained
      """
      # TODO: Implement

      # Hyperparameters
      learning_rate = 0.005
      
      # TODO: Build the layers
      model = Sequential()
      model.add(Embedding(english_vocab_size, 256, input_length=input_shape[1], input_shape=input_shape[1:]))
      model.add(GRU(256, return_sequences=True))    
      model.add(TimeDistributed(Dense(1024, activation='relu')))
      model.add(Dropout(0.5))
      model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax'))) 

      # Compile model
      model.compile(loss=sparse_categorical_crossentropy,
                    optimizer=Adam(learning_rate),
                    metrics=['accuracy'])
      return model


  # Reshaping the input to work with a basic RNN
  tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
  tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))


  simple_rnn_model = embed_model(
      tmp_x.shape,
      preproc_french_sentences.shape[1],
      len(english_tokenizer.word_index)+1,
      len(french_tokenizer.word_index)+1)


  # Here we start to train the model and pass the english text and the max_sequence_length, with vocab size for both english and french text
  history=simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.2)

  simple_rnn_model.save('model.h5')
