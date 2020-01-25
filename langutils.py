import nltk
import string
from spellchecker import SpellChecker

def remove_punctuation(text):
    punctuation = string.punctuation + "".join(["\n", " "])
    for symbol in punctuation:
        if symbol in text:
            text.remove(symbol)
    return text

def get_word_and_sentence_count(paragraph):
    word_count = paragraph.count(" ") + 1
    sentence_count = paragraph.count(".")
    return word_count, sentence_count

def get_correct_to_total_ratio(paragraph):
    paragraph = paragraph.translate(str.maketrans("", "", string.punctuation))
    all_words = paragraph.split(" ")

    spell = SpellChecker()
    misspelled = spell.unknown(all_words)
    misspelled = remove_punctuation(misspelled)

    correct_words = len(all_words) - len(misspelled)

    return correct_words / len(all_words)

def clean_sentence(answer):
    stop_words = set(nltk.corpus.stopwords.words('english'))

    cleaned_sentence = []
    sentences = nltk.sent_tokenize(answer)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        clean_words = [
            word.lower()
            for word in words
            if len(word) > 1 and word not in stop_words
        ]

        tagged_words = nltk.pos_tag(clean_words)
        proper_words = [
            tagged_word[0]
            for tagged_word in tagged_words
            if tagged_word[1][1] in {'J', 'N', 'R', 'V'}
        ]
        cleaned_sentence.extend(proper_words)

    return cleaned_sentence

def get_synonyms(word):
    synonyms = set()
    synsets = nltk.corpus.wordnet.synsets(word)
    for synset in synsets[:5]:
        synonym = synset.lemmas()[0].name()
        synonyms.add(synonym)

    return synonyms

def prepare_vector(answer):
    vector = dict()

    words, sentences = get_word_and_sentence_count(answer)
    correct_ratio = get_correct_to_total_ratio(answer)

    vector['word_count'] = words
    vector['sentence_count'] = sentences
    vector['correct_ratio'] = correct_ratio

    sentence = clean_sentence(answer)

    fdist = nltk.FreqDist(sentence)
    word_freq = dict(fdist)

    vector['clean_words'] = len(word_freq)

    for word, freq in word_freq.items():
        vector[word] = freq
        for synonym in get_synonyms(word):
            vector[synonym] = freq

    return vector

if __name__ == "__main__":
    SAMPLE_SENTENCE = "Hello, I am Ajay Raj. I write code. Code is addictive."
    # sentence = clean_sentence(SAMPLE_SENTENCE)
    # print(sentence)

    # synonyms = get_synonyms("good")
    # print(synonyms)

    vector = prepare_vector(SAMPLE_SENTENCE)
    print(vector)
