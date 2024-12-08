from pyroaring import BitMap
from text_processor import ITextProcessor, TextProcessor


class PositionalIndex:
    def __init__(self, max_word_delta: int = 3, text_processor: ITextProcessor = TextProcessor()):
        self.max_word_delta = max_word_delta
        self.word2doc = {}
        self.wordndoc2poses = {}
        self._last_doc_id = 0
        self._text_processor = text_processor

    def add(self, text: str):
        words = self._text_processor.preprocess_text(text)
        for pos, word in enumerate(words):
            self._add_word(word, self._last_doc_id, pos)
        self._last_doc_id += 1

    def search(self, phrase: str) -> BitMap:
        words = self._text_processor.preprocess_text(phrase)
        if not words:
            return BitMap()

        docs_matches = self._get_documents_matches(words)
        return self._search_phrase_in_documents(words, docs_matches)

    def _add_word(self, word: str, doc_id: int, position: int):
        self.word2doc.setdefault(word, BitMap()).add(doc_id)
        self.wordndoc2poses.setdefault((word, doc_id), BitMap()).add(position)

    def _get_documents_matches(self, words: list[str]) -> BitMap:
        docs_matches = self.word2doc.get(words[0], BitMap()).copy()
        for word in words[1:]:
            docs_matches &= self.word2doc.get(word, BitMap())
            if docs_matches == BitMap():
                return BitMap()
        return docs_matches

    def _search_phrase_in_documents(self, words: list[str], docs_matches: BitMap) -> BitMap:
        result = BitMap()
        for doc_id in docs_matches:
            word2poses = [sorted(self.wordndoc2poses[(word, doc_id)]) for word in words]
            if self._search_phrase_in_positions(word2poses):
                result.add(doc_id)
        return result

    def _search_phrase_in_positions(self, word2poses: list[list[int]]) -> bool:
        indices = [0] * len(word2poses)
        while indices[0] < len(word2poses[0]):
            if self._check_cur_indices_for_search_in_positions(word2poses, indices):
                return True
            indices[0] += 1
        return False

    def _check_cur_indices_for_search_in_positions(self, word2poses: list[list[int]], indices: list[int]) -> bool:
        prev_word_pos = word2poses[0][indices[0]]

        for i in range(1, len(word2poses)):
            positions = word2poses[i]
            index_of_cur_position = indices[i]

            while index_of_cur_position < len(positions) and positions[index_of_cur_position] < prev_word_pos:
                index_of_cur_position += 1

            if index_of_cur_position == len(positions):
                return False

            cur_word_pos = positions[index_of_cur_position]
            if cur_word_pos - prev_word_pos > self.max_word_delta:
                return False

            indices[i] = index_of_cur_position
            prev_word_pos = cur_word_pos

        return True
