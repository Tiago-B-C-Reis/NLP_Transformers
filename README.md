# NLP_Transformers 
![Screenshot from 2023-03-03 10-42-12.png](Screenshot%20from%202023-03-03%2010-42-12.png)
# Pipeline function from HuggingFace.

There are 3 types of models:
- Decoders (ex: BERT)
- Encoders (ex: GPT)
- Sequence-to-sequence (ex: T5)

All of those have in common to be trained using the transformers architecture.
This project uses the "pipeline" functions from HuggingFace transformers that allow us to use these basic 
actions by using the correct checkpoints:
- [Mask-filling](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L91-L96):
  - Chosen model = [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)
- [Text-classification](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L97-L102):
  - Chosen model = [distilbert-base-uncased-finetuned-sst-2-english ](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [Zero-shot text-classification](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L103-L111): 
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


