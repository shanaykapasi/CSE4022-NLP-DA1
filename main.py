import nltk.corpus
from collections import Counter
nltk.download('brown')

# Question 1
# The Brown corpus:
brown = nltk.corpus.brown

word_count = dict(sorted(Counter(brown.words()).items(), key=lambda item: item[1], reverse=True))
unique_words = list(word_count.keys())

print("Size of word tokens, ", word_count["tokens"])

print("Size of word types, ", word_count["types"])

print("Size of category government, ", len(brown.tagged_paras(categories='government')))

[print(f"No. {i+1} most frequent word, {unique_words[i]}") for i in range(10)]

print("Number of sentences, ",len(brown.sents()))

# Question 2

from nltk.corpus import indian
nltk.download('indian')
print(indian.words())
print(indian.tagged_words())