from transformers import pipeline, set_seed
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer


# --------------------------------------------------------------------------------------------------------------------
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
        label = input("Please add a classification label: (ex:Education, Politics, Travel)\n")
        classification_labels.append(label)
        break_point = input("Do you want to add more?(yes/no) ")
        if break_point == "no":
            break

    return classification_labels


def language_selection():
    """This function will allow the user to translate the text from english to 'x' language available."""
    lang_sel = str(input(
        "To what language do you want to translate the text to?\n"
        " 1 - French;\n"
        " 2 - Spanish;\n"
        " 3 - German;\n"
        " 4 - Japanese.\n"
    ))

    model_selected = ""

    if lang_sel == "1":
        model_selected = "Helsinki-NLP/opus-mt-en-fr"
    elif lang_sel == "2":
        model_selected = "Helsinki-NLP/opus-mt-en-es"
    elif lang_sel == "3":
        model_selected = "Helsinki-NLP/opus-mt-en-de"
    elif lang_sel == "4":
        model_selected = "Helsinki-NLP/opus-mt-en-jap"

    return model_selected

# --------------------------------------------------------------------------------------------------------------------


flag = True
while flag:
    text_input = input("Please insert the text to be processed (english only): ")
    # Text-classification pipeline.
    classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    output_0 = classifier(text_input)

    if output_0[0]["label"] == "en":
        while True:
            task_deployer = str(input(
                "Please choose one of the available options for text processing (enter nº between 1 and 8):\n"
                " 1 - Mask-filling\n"
                " 2 - Text-classification\n"
                " 3 - Zero-shot text-classification\n"
                " 4 - Text generation\n"
                " 5 - Token classification\n"
                " 6 - Question answering\n"
                " 7 - Summarization\n"
                " 8 - Translation\n"
            ))

            if task_deployer == "1":
                # Mask-filling pipeline.
                unmasker = pipeline('fill-mask', model='distilbert-base-cased')
                output_1 = unmasker(mask_filling(text_input))
                print(output_1)
                break
            elif task_deployer == "2":
                # Text-classification pipeline.
                classifier = pipeline("sentiment-analysis")
                output_2 = classifier(text_input)
                print(output_2)
                break
            elif task_deployer == "3":
                # Zero-shot text-classification pipeline.
                zero_shot_classifier = pipeline("zero-shot-classification")
                output_3 = zero_shot_classifier(
                    text_input,
                    candidate_labels=zero_shot()
                )
                print(output_3)
                break
            elif task_deployer == "4":
                # Text generation pipeline.
                generator = pipeline('text-generation', model='gpt2')
                set_seed(42)
                output_4 = generator(
                    text_input,
                    max_length=50,
                    num_return_sequences=5
                )
                print(output_4)
                break
            elif task_deployer == "5":
                # Token classification pipeline.
                token_class = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english",
                                       aggregation_strategy="simple")
                output_5 = token_class(text_input)
                print(output_5)
                break
            elif task_deployer == "6":
                # Question answering pipeline.
                model_name = "deepset/roberta-base-squad2"
                nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
                question = input("Please give a question related to the given text:\n")
                QA_input = {
                    'question': question,
                    'context': text_input
                }
                output_6 = nlp(QA_input)
                print(output_6)
                break
            elif task_deployer == "7":
                # Summarization pipeline.
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                output_7 = summarizer(text_input)
                print(output_7)
                break
            elif task_deployer == "8":
                # Translation pipeline.
                translator = pipeline("translation", model=str(language_selection()))
                output_8 = translator(text_input)
                print(output_8)
                break
            else:
                print("Please enter a number between 1 and 8!")
        flag = False
    else:
        print("Please enter an english string.")
