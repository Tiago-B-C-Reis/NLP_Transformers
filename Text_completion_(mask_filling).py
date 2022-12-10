# !pip install datasets evaluate transformers[sentencepiece]
# pip install transformers
# pip install transformers[tf-cpu]
from transformers import pipeline
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

text = "The game 's battle system , the BliTZ system , is carried over directly from Valkyira Chronicles ."

def mask_fill(text):
    """This function runs an NLP algorithm for fill-mask for a given input text and an input token to mask. """

    # this lines of code are just to split the text in the tokens that the algorithm is going to use, in case it can
    # be useful.
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    sequence = text
    tokens = tokenizer.tokenize(sequence)

    mask_token = input(print("For each word do you want to receive a replacing suggestion?"))

    if mask_token in text:
        # replaces the existing string given in "mask_token" inside the string "text" with "[MASK]":
        new_text = text.replace(mask_token, "[MASK]")
        print(new_text)

        if isinstance(new_text, str):
            print('Type of variable is string')
        else:
            print('Type is variable is not string')

        unmasker = pipeline('fill-mask', model='distilbert-base-cased')
        unmasker(new_text)
    else:
        print("That string does not exist inside the given text!")

mask_fill(text)
