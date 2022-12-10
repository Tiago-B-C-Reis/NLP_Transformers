# !pip install datasets evaluate transformers[sentencepiece]
# pip install transformers
# pip install transformers[tf-cpu]
from transformers import pipeline
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
sequence = " The game 's battle system , the BliTZ system , is carried over directly from Valkyira Chronicles . "
tokens = tokenizer.tokenize(sequence)
print(tokens)

# Result:
# ['The', 'game', "'", 's', 'battle', 'system', ',', 'the', 'B', '##li', '##T', '##Z', 'system', ',', 'is',
# 'carried', 'over', 'directly', 'from', 'Val', '##ky', '##ira', 'Chronicles', '.']

unmasker = pipeline('fill-mask', model='distilbert-base-cased')
unmasker(" The game 's battle [MASK] , the BliTZ system , is carried over directly from Valkyira Chronicles . "
         "During missions , players select each unit using a top @-@ down perspective of the battlefield map : "
         "once a character is selected , the player moves the character around the battlefield in third @-@ person . "
         "A character can only act once per @-@ turn , but characters can be granted multiple turns at the expense of "
         "other characters ' turns . "
         "Each character has a field and distance of movement limited by their Action Gauge . "
         "Up to nine characters can be assigned to a single mission . During gameplay , characters will call out if "
         "something happens to them , such as their health points ( HP ) getting low or being knocked out by "
         "enemy attacks . "
         "Each character has specific \" Potentials \" , skills unique to each character . "
         "They are divided into \" Personal Potential \" , which are innate skills that remain unaltered unless "
         "otherwise dictated by the story and can either help or impede a character , and \" Battle Potentials \" , "
         "which are grown throughout the game and always grant boons to a character . To learn Battle Potentials , "
         "each character has a unique \" Masters Table \" , a grid @-@ based skill table that can be used to acquire "
         "and link different skills . Characters also have Special Abilities that grant them temporary boosts on the "
         "battlefield : Kurt can activate \" Direct Command \" and move around the battlefield without depleting his "
         "Action Point gauge , the character Reila can shift into her \" Valkyria Form \" and become invincible , "
         "while Imca can target multiple enemy units with her heavy weapon . \n ")


# Steps that I have performed:
# 1 - Learn from the HuggingFace course.

# 2 - Follow the transformers installation steps "https://huggingface.co/docs/transformers/installation".

# 3 - Download 'wikitext-2-raw-v1' with only the train split, read it with a json viewer and take
# the 11th example (row_idx : 10).

# 4 - Use the tokenizer from the "distilbert-base-cased" model checkpoint (this model uses the "WordPiece" tokenization)
# and find the token nº6 (index : 5), that is the token "battle".

# 5 - From transformers import the pipeline. This pipeline already perform all the needed steps starting from
# tokenizing, using the model and post-processing it in other to get the predictions.

# 6 - Used the requested "distilbert-base-cased" model (that is a checkpoint) based on the BERT architecture.
# Since this model is based on the BERT architecture, it means that is mainly an "Encoder" and one of its main tasks
# is to do "Masked language modeling" which is the required purpose. In the model description is mentioned that I
# can use the raw model for masked language modeling, which means that the model can perform a good result without the
# need for fine-tuning in this particular case.

# 7 - Input the 11th example from the 'wikitext-2-raw-v1' on the "unmasker()", but with the [MASK] token replacing the
# token nº6 that was 'battle'.

# 8 - Run the code.

# 9 - Result:
# Nº1 with the score of 0.19702 = token_str: 'mechanic'.
