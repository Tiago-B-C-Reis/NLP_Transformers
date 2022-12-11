# !pip install datasets evaluate transformers[sentencepiece]
# pip install transformers
# pip install transformers[tf-cpu]
from transformers import pipeline, set_seed
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

text_input = input("Text: ")


def mask_filling(text_input):
    """This function runs an NLP algorithm for fill-mask for a given input text and an input token to mask. """

    # this lines of code are just to split the text in the tokens that the algorithm is going to use, in case it can
    # be useful.
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    sequence = text_input
    tokens = tokenizer.tokenize(sequence)

    while True:
        mask_token = input("For each word do you want to receive a replacing suggestion?\n")
        try:
            if mask_token in text_input:
                # replaces the existing string given in "mask_token" inside the string "text" with "[MASK]":
                new_text = text_input.replace(mask_token, "[MASK]")
                return new_text
            else:
                print("That string does not exist inside the given text!")
        except ValueError:
            print("That input is not acceptable!\n")


def zero_shot():
    """This function allows the user to decide on which classification the user wants to classify the text"""
    classification_labels = []

    while True:
        label = input("Please add a classification label: (ex:Education, Politics, Travel)")
        classification_labels.append(label)
        break_point = input("Do you want to add more?(yes/no) ")
        if break_point == "no":
            break

    return classification_labels

# --------------------------------------------------------------------------------------------------------------------


# Mask-filling pipeline.
unmasker = pipeline('fill-mask', model='distilbert-base-cased')
output = unmasker(mask_filling(text_input))
print(output)

# Text-classification pipeline.
classifier = pipeline("sentiment-analysis")
output_1 = classifier(text_input)
print(output_1)

# Zero-shot text-classification pipeline.
zero_shot_classifier = pipeline("zero-shot-classification")
output_2 = zero_shot_classifier(
    text_input,
    candidate_labels=zero_shot()
)
print(output_2)

# Text generation pipeline.
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
output_3 = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print(output_3)


