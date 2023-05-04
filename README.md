# NLP_Transformers 
![Screenshot from 2023-03-03 10-42-12.png](Screenshot%20from%202023-03-03%2010-42-12.png)
# Pipeline function from [HuggingFace](https://huggingface.co/docs/transformers/index).

There are 3 types of models:
- Decoders (ex: BERT)
- Encoders (ex: GPT)
- Sequence-to-sequence (ex: T5)

All of those have in common to be trained using the transformers architecture.
This project uses the "pipeline" functions from HuggingFace transformers that allow us to use these basic 
actions by using the correct checkpoints:
- [Text completion - Mask-filling](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L91-L96):
  - Chosen model = [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)
- [Text-classification - sentiment analysis](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L97-L102):
  - Chosen model = [distilbert-base-uncased-finetuned-sst-2-english ](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [Zero-shot text-classification](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L103-L111):
  - Chosen model = [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli )
- [Text generation](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L112-L122):
  - Chosen model = [gpt2](https://huggingface.co/gpt2)
- [Token classification](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L123-L129):
  - Chosen model = [xlm-roberta-large-finetuned-conll03-english](https://huggingface.co/xlm-roberta-large-finetuned-conll03-english)
- [Question answering](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L130-L141):
  - Chosen model = [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
- [Summarization](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L142-L147):
  - Chosen model = [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
- [Translation](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L148-L153):
  - Chosen model:
    - [Helsinki-NLP/opus-mt-en-fr - (English to French)](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr)
    - [Helsinki-NLP/opus-mt-en-es - (English to Spanish)](https://huggingface.co/Helsinki-NLP/opus-mt-en-es)
    - [Helsinki-NLP/opus-mt-en-de - (English to German)](https://huggingface.co/Helsinki-NLP/opus-mt-de-en)
    - [Helsinki-NLP/opus-mt-en-jap - (English to Japanese)](https://huggingface.co/Helsinki-NLP/opus-mt-en-jap)


### 1- Text completion - Mask-filling:
- This task involves filling in a blank or missing word in a sentence or paragraph using a pre-trained language model. The process is also known as Mask-filling or Cloze Test.

### 2- Text-classification - sentiment analysis: 
- This task involves classifying a given text as expressing a positive, negative, or neutral sentiment. Sentiment analysis is commonly used in social media monitoring, brand monitoring, and customer service.

### 3- Zero-shot text-classification: 
- This task involves classifying a given text into pre-defined categories, even if the text has not been seen before by the model. This approach enables the model to generalize to new and unseen categories.

### 4- Text generation: 
- This task involves generating new text based on a given prompt or context. The generated text can be used for a variety of applications, such as chatbots, language translation, and creative writing.

### 5- Token classification: 
- This task involves labeling each token (word or subword) in a given text with a specific tag, such as part-of-speech tags or named entity tags. This approach is used in information extraction, named entity recognition, and machine translation.

### 6- Question answering: 
- This task involves answering a question based on a given context or passage. The model is trained to understand the context and generate an answer that is relevant to the question.

### 7- Summarization: 
- This task involves creating a shorter version of a longer text while preserving its most important information. Summarization is commonly used in news articles, academic papers, and legal documents.

### 8- Translation: 
- This task involves translating a given text from one language to another. The model is trained to understand the meaning of the source text and generate a translation that accurately conveys its meaning in the target language.
