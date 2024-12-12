import json
from datasets import load_dataset

"""
Returns an array of strings
"""
def fetch_messages():
    """
    with open('messages.json', 'r') as openfile:
        json_object = json.load(openfile)

    messages = []
    for message in json_object["messages"]:
        messages.append(message)

    denom = 1
    messages = messages[:len(messages)//denom]
    """

    messages = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")

    return messages['text']

"""
TRAIN, SAVE TOKENIZER
"""
from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.trainers import BpeTrainer

def build_tokenizer(messages, vocab_size=1000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)

    tokenizer.train_from_iterator(messages, trainer=trainer)

    from tokenizers import decoders
    tokenizer.decoder = decoders.BPEDecoder()

    tokenizer.save("tokenizer.json")
    print("saved tokenizer!!!")

    return tokenizer

def encode_messages(messages, tokenizer, eos):
    encoded = []
    length = 0 
    for message in messages:
        enc = tokenizer.encode(message).ids
        #enc.extend([eos])  
        encoded.extend(enc)
        length += 1 
        if length>1000:
            break

    return encoded

"""
seeing how to pull a better dataset from huggingface
"""
#if __name__ == "__main__":
  #ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
  #messages = fetch_messages()