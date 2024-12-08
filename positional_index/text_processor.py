import nltk
from nltk.stem import PorterStemmer
from abc import ABC, abstractmethod

nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")

class ITextProcessor(ABC):
    @abstractmethod
    def preprocess_text(self, text: str, lang: str = "english") -> list[str]:
        pass

class TextProcessor:
    def __init__(
        self,
        language: str = "english",
        stemmer=PorterStemmer(),
        stop_words: set = set()
    ):
        self.language = language
        self.stemmer = stemmer
        self.stop_words = stop_words
        self.stop_words.update(nltk.corpus.stopwords.words(language))

    def preprocess_text(self, text: str) -> list[str]:
        words = nltk.tokenize.word_tokenize(text)
        return [self.stemmer.stem(word) for word in words if word.isalnum() and word.lower() not in self.stop_words]
