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
- Mask-filling:
  - Model used = [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)
  - [Code link](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L91-L96)
- Text-classification:
  - Model used = [distilbert-base-uncased-finetuned-sst-2-english ](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
  - [Code link](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L97-L102)
- Zero-shot text-classification: 
  - [Code link](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L103-L111)
- Text generation:
  - Model used = [gpt2](https://huggingface.co/gpt2)
  - [Code link](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L112-L122)
- Token classification:
  - Model used = [xlm-roberta-large-finetuned-conll03-english](https://huggingface.co/xlm-roberta-large-finetuned-conll03-english)
  - [Code link](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L123-L129)
- Question answering:
  - Model used = [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
  - [Code link](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L130-L141)
- Summarization:
  - Model used = [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
  - [Code link](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L142-L147)
- Translation:
  - Model used:
    - [Helsinki-NLP/opus-mt-en-fr - (English to French)](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr)
    - [Helsinki-NLP/opus-mt-en-es - (English to Spanish)](https://huggingface.co/Helsinki-NLP/opus-mt-en-es)
    - [Helsinki-NLP/opus-mt-en-de - (English to German)](https://huggingface.co/Helsinki-NLP/opus-mt-de-en)
    - [Helsinki-NLP/opus-mt-en-jap - (English to Japanese)](https://huggingface.co/Helsinki-NLP/opus-mt-en-jap)
  - [Code link](https://github.com/Tiago-B-C-Reis/NLP_Transformers/blob/c6dc674c14153db0133982ff9bb51373ad81b44d/Transformers.py#L148-L153)


