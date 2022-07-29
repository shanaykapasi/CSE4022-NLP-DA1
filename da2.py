import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

corpus_raw = ""
stop_words = list(set(stopwords.words('english')))

with open("corpus.txt", "r") as f:
    corpus_raw = f.read()

# Sentence Segmentation
sentences = sent_tokenize(corpus_raw)
print("\n")
[print(i) for i in sentences]
print("\n")
input("Press Enter to Continue...")

# Word Segmentation
words = word_tokenize(corpus_raw)
print("\n")
[print(i) for i in words]
print("\n")
input("Press Enter to Continue...")

# Convert to Lowercase
sentences = list(map(lambda x: x.lower(), sentences))
words = list(map(lambda x: x.lower(), words))
print("\n")
[print(i) for i in sentences]
print("\n")
input("Press Enter to Continue...")
[print(i) for i in words]
print("\n")
input("Press Enter to Continue...")

# Stop Words Removal
non_stop_words = [w for w in words if not w in stop_words]
print("\n")
[print(i) for i in non_stop_words]
print("\n")
input("Press Enter to Continue...")

# Stemming
ps = PorterStemmer()
stemmed_non_stop_words = [ps.stem(w) for w in non_stop_words]
print("\n")
[print(i) for i in stemmed_non_stop_words]
print("\n")
input("Press Enter to Continue...")

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_stemmed_non_stop_words = [
    lemmatizer.lemmatize(w) for w in stemmed_non_stop_words]
print("\n")
[print(i) for i in lemmatized_stemmed_non_stop_words]
print("\n")
input("Press Enter to Continue...")

# POS Tagging
tagged = nltk.pos_tag(lemmatized_stemmed_non_stop_words)
print("\n")
[print(i) for i in tagged]
print("\n")
input("Press Enter to Continue...")
