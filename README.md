# NLP_Transformers 
# Pipeline function from HuggingFace.

There are 3 types of models:
- Decoders (ex: BERT)
- Encoders (ex: GPT)
- Sequence-to-sequence (ex: T5)

All of those have in common to be trained using the transformers architecture.
This project uses the "pipeline" functions from HuggingFace transformers that allow us to use these basic 
actions by using the correct checkpoints:
- Text classification;
- Zero-shot classification;
- Text generation;
- Text completion (mask filling)
- Token classification;
- Question answering;
- Summarization;
- Translation.

