from collections import Counter

import nltk
import pyphen
import regex as re


def _split_words_from_text(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"(\b\p{L}+\b)", text)]


def _estimate_number_of_syllables_in_word_pyphen(word: str) -> int:
    dic = pyphen.Pyphen(lang="en")
    return len(dic.positions(word)) + 1


def _syllables_per_word(words: list[str]) -> float:
    return sum(
        _estimate_number_of_syllables_in_word_pyphen(word) * freq
        for word, freq in Counter(words).items()
    ) / len(words)


def fleisch_readability(text: str) -> float:
    words = _split_words_from_text(text)
    syllables_per_word = _syllables_per_word(words)
    num_sentences = len(nltk.sent_tokenize(text))
    sentence_length = len(words) / num_sentences
    return 206.835 - sentence_length * 1.015 - syllables_per_word * 84.6
