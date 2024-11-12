import os
import statistics

import nltk
import regex as re

from bachelor.dataset import load_dataset


def split_words_from_text(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"(\b\p{L}+\b)", text)]


def get_sentence_length(texts: list[str]):
    word_counts = map(lambda x: len(split_words_from_text(x)), texts)
    word_lengths = map(lambda x: statistics.fmean(map(len, split_words_from_text(x))), texts)
    sentence_counts = map(lambda x: len(nltk.sent_tokenize(x)), texts)
    sentence_lengths = map(lambda x: x[0] / x[1], zip(word_counts, sentence_counts))
    return {"sentence_len": list(sentence_lengths), "word_len": list(word_lengths)}


def main() -> None:
    dataset = load_dataset(f"{os.path.dirname(os.path.dirname(__file__))}/data/ef_POStagged_orig_corrected.csv")
    dataset = dataset.map(get_sentence_length, batched=True, input_columns=["text"])
    print("Median sentence length:", statistics.median(dataset["sentence_len"]))
    print("Mean sentence length:", statistics.fmean(dataset["sentence_len"]))
    print("Median word length:", statistics.median(dataset["word_len"]))
    print("Mean word length:", statistics.fmean(dataset["word_len"]))


if __name__ == "__main__":
    main()
