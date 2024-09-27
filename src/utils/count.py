import nltk

def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    num_words = len(tokens)
    # print(tokens)
    return num_words


def count_tokens(tokenizer, text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    return inputs.shape[1]

if __name__ == "__main__":
    pass