
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'you love my dog!',
    'Do you think my dog is amazing?'
]

# tokenize = Tokenizer(num_words = 10)
tokenize = Tokenizer(oov_token = "<OOV>")
tokenize.fit_on_texts(sentences)

word_index = tokenize.word_index
print(word_index)

sequence = tokenize.texts_to_sequences(sentences)
print(sequence)

test_data = [
    'i really love my dog',
    'my dog loves manatee'
]
test_seq = tokenize.texts_to_sequences(test_data)
print(test_seq)
word_index = tokenize.word_index
print(word_index)