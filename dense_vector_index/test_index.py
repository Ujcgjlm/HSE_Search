import pytest
import random
import numpy as np
from unittest.mock import MagicMock
from sklearn.decomposition import PCA
import faiss

from dense_vector_index import DenseVectorIndex

class MockEmbedder:
    def __init__(self):
        pass

    def get_embeddings(self, texts):
        return np.array([np.random.rand(768) for _ in texts])

@pytest.fixture
def dense_vector_index():
    index = DenseVectorIndex(n_components=2)
    return index

@pytest.fixture
def dense_vector_index_big():
    index = DenseVectorIndex()
    return index

def test_add_texts(dense_vector_index):
    dense_vector_index.add("Test sentence one.")
    dense_vector_index.add("Test sentence two.")
    
    assert len(dense_vector_index.texts) == 2

def test_build_index(dense_vector_index):
    dense_vector_index.add("Test sentence one.")
    dense_vector_index.add("Test sentence two.")
    dense_vector_index.build_index()
    
    assert len(dense_vector_index.embeddings) == 2
    assert dense_vector_index.index is not None

def test_search(dense_vector_index):
    dense_vector_index.add("Hello world")
    dense_vector_index.add("Hi, how are you?")
    dense_vector_index.add("Good day")
    dense_vector_index.build_index()

    results = dense_vector_index.search("Hello", k=2)
    
    assert isinstance(results, list)
    assert len(results) >= 1

def test_no_results_search(dense_vector_index):
    dense_vector_index.add("Hello world")
    dense_vector_index.add("Hi, how are you?")
    dense_vector_index.add("Good day")
    dense_vector_index.build_index()

    results = dense_vector_index.search("No match here", k=2)

    assert len(results) > 0

def generate_random_text_with_phrase(phrase, num_words=5):
    random.seed(42)
    words = ["the", "and", "to", "of", "a", "in", "that", "is", "it", "with", "as", "for",
             "was", "on", "are", "by", "this", "at", "or", "which",
             "time", "year", "people", "way", "day", "man", "thing", "woman", "life", "child", 
             "world", "school", "state", "family", "student", "group", "country", "problem", 
             "hand", "part", "place", "case", "week", "company", "system", "program", "question",
             "work", "government", "number", "night", "point", "home", "water", "room", "mother",
             "area", "money", "story", "fact", "month", "lot", "right", "study", "book", "eye", 
             "job", "word", "business", "issue", "side", "kind", "head", "house", "service", "friend"]
    
    text = [random.choice(words) for _ in range(num_words)]
    insertion_point = random.randint(1, num_words - 2)
    text.insert(insertion_point, phrase)
    return " ".join(text)

def test_add_large_documents(dense_vector_index_big):
    phrases = [
        "climate change impacts",
        "artificial intelligence technologies",
        "quantum computing advancements",
        "space exploration missions",
        "renewable energy sources",
    ]

    number_of_texts_with_phrase = 30
    large_texts = [generate_random_text_with_phrase(phrase) for phrase in phrases * number_of_texts_with_phrase]

    for text in large_texts:
        dense_vector_index_big.add(text)

    dense_vector_index_big.build_index()

    for phrase in phrases:
        results = dense_vector_index_big.search(phrase, k=5)
        assert all([phrase in result for result in results])
