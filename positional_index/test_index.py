import pytest
import random
from pyroaring import BitMap
from text_processor import TextProcessor
from positional_index import PositionalIndex

@pytest.fixture
def text_processor():
    return TextProcessor()

@pytest.fixture
def positional_index(text_processor):
    return PositionalIndex(max_word_delta=3, text_processor=text_processor)

def test_add_single_doc(positional_index):
    text = "This python is not great"
    positional_index.add(text)
    assert positional_index._last_doc_id == 1

    assert "python" in positional_index.word2doc
    assert "great" in positional_index.word2doc
    # Stop words
    assert "this" not in positional_index.word2doc
    assert "is" not in positional_index.word2doc
    assert "not" not in positional_index.word2doc

def test_search_found_single_word(positional_index):
    text = "Python is great"
    positional_index.add(text)
    result = positional_index.search("Python")
    assert isinstance(result, BitMap)
    assert 0 in result

def test_search_found_phrase(positional_index):
    text = "Python is great and Python is powerful"
    positional_index.add(text)
    result = positional_index.search("Python is powerful")
    assert 0 in result

def test_search_phrase_with_stop_words_found(positional_index):
    text = "Python is great"
    positional_index.add(text)
    result = positional_index.search("Python is not great")
    assert 0 in result

def test_search_phrase_not_found(positional_index):
    text = "Python is great"
    positional_index.add(text)
    result = positional_index.search("Python is horrible")
    assert len(result) == 0

def test_phrase_out_of_word_delta(positional_index):
    text = "Python supports a concept of iteration over containers"
    positional_index.add(text)
    result = positional_index.search("Python containers")
    assert len(result) == 0

def test_add_multiple_docs(positional_index):
    texts = [
        "Python is great",
        "Java is versatile",
        "C++ is powerful",
        "Python and Java both are popular"
    ]
    for text in texts:
        positional_index.add(text)
    assert positional_index._last_doc_id == len(texts)

def test_search_across_multiple_docs(positional_index):
    texts = [
        "Python is great",
        "Python is powerful",
        "Java is versatile"
    ]
    for text in texts:
        positional_index.add(text)
    result = positional_index.search("Python")
    assert len(result) == 2
    assert 0 in result
    assert 1 in result

def test_empty_text(positional_index):
    empty_text = ""
    positional_index.add(empty_text)
    result = positional_index.search("")
    assert len(result) == 0

def test_nonexistent_phrase(positional_index):
    text = "Python is great"
    positional_index.add(text)
    result = positional_index.search("Ruby is great")
    assert len(result) == 0

def generate_random_text_with_phrase(phrase, num_words=1000):
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

def test_add_large_documents(positional_index):
    phrases = [
        "climate change",
        "artificial intelligence",
        "quantum computing",
        "space exploration",
        "renewable energy",
    ]

    number_of_texts_with_phrase = 10
    large_texts = [generate_random_text_with_phrase(phrase) for phrase in phrases * number_of_texts_with_phrase]

    for text in large_texts:
        positional_index.add(text)

    for phrase in phrases:
        result = positional_index.search(phrase)
        assert len(result) == number_of_texts_with_phrase
