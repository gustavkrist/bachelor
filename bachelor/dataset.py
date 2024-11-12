import random
from typing import TypedDict

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Value,
)
from nltk import PunktTokenizer
from pandas import DataFrame
from transformers import AutoTokenizer

from bachelor.utils import level_to_cefr


def thin_dataset(dataset: Dataset) -> Dataset:
    ds_cefr = dataset.map(lambda x: {"cefr": [level_to_cefr(y) for y in x["level"]]}, batched=True)
    a1_rows = list(filter(lambda x: x[1] == "A1", enumerate(ds_cefr["cefr"])))
    a2_rows = list(filter(lambda x: x[1] == "A2", enumerate(ds_cefr["cefr"])))
    b1_rows = list(filter(lambda x: x[1] == "B1", enumerate(ds_cefr["cefr"])))
    b2_rows = list(filter(lambda x: x[1] == "B2", enumerate(ds_cefr["cefr"])))
    c1_rows = list(filter(lambda x: x[1] == "C1", enumerate(ds_cefr["cefr"])))
    c2_rows = list(filter(lambda x: x[1] == "C2", enumerate(ds_cefr["cefr"])))
    indices: list[int] = []
    random.seed(0)
    for rows, sample_size in ((a1_rows, 0.1), (a2_rows, 0.2), (b1_rows, 0.4), (b2_rows, 1)):
        indices.extend(map(lambda x: x[0], random.sample(rows, k=int(len(rows) * sample_size))))
    indices.extend(map(lambda x: x[0], c1_rows))
    indices.extend(map(lambda x: x[0], c2_rows))
    return dataset.select(indices)


def load_dataset(
    path: str,
    features: Features | None = None,
    cols: list[str] | None = None,
    thin: bool = True,
) -> Dataset:
    "Loads dataset from specified path and created train/test splits"
    if features is None:
        features = Features(
            {
                "level": ClassLabel(num_classes=16, names=[str(i) for i in range(1, 17)]),
                "grade": Value("int32"),
                "corrected": Value("string"),
            }
        )
    if cols is None:
        cols = ["level", "grade", "corrected"]
    dataset = Dataset.from_csv(path, usecols=cols)
    assert isinstance(dataset, Dataset)
    if thin:
        dataset = thin_dataset(dataset)
    dataset = dataset.rename_columns({"level": "label", "corrected": "text"}).shuffle(seed=0)
    return dataset


def split_dataset(dataset: Dataset, train_size: float = 0.8) -> DatasetDict:
    train_test_splits = dataset.train_test_split(train_size=train_size, seed=0)
    evaluate_test_splits = train_test_splits["test"].train_test_split(train_size=0.5, seed=0)
    return DatasetDict(
        {
            "train": train_test_splits["train"],
            "eval": evaluate_test_splits["train"],
            "test": evaluate_test_splits["test"],
        }
    )


def filter_dataset(dataset: DatasetDict, minimum_grade: int | None = None) -> DatasetDict:
    if minimum_grade is not None:
        dataset = dataset.filter(
            lambda x: (grade > minimum_grade for grade in x["grade"]), batched=True
        )
    return dataset.remove_columns("grade")


def tokenize_dataset(dataset: DatasetDict) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-cased", clean_up_tokenization_spaces=False
    )

    def tokenize_function(rows):  # type: ignore
        return tokenizer(rows["text"], padding="max_length", truncation=True)

    return dataset.map(tokenize_function, batched=True)


def get_dataset(path: str, thin: bool = True) -> DatasetDict:
    return tokenize_dataset(filter_dataset(split_dataset(load_dataset(path, thin=thin))))


def create_windowed_dataset(input_path: str, output_path: str) -> None:
    class DatasetRows(TypedDict):
        text: list[str]
        label: list[int]
        grade: list[int]

    dataset = load_dataset(input_path)
    tokenizer = PunktTokenizer("english")

    def sliding_window_sentences(text: str, window_size: int = 150) -> list[str]:
        windows = []
        for sent_start, _ in tokenizer.span_tokenize(text):
            windows.append(text[sent_start : sent_start + window_size])
        return windows

    def sliding_window_dataset(rows: DatasetRows) -> DatasetRows:
        res: DatasetRows = {"text": [], "label": [], "grade": []}
        for i in range(len(rows["text"])):
            windows = sliding_window_sentences(rows["text"][i])
            for k in res:
                if k != "text":
                    res[k].extend([rows[k][i]] * len(windows))  # type: ignore
            res["text"].extend(windows)
        return res

    dataset = dataset.map(
        sliding_window_dataset,
        batched=True,
        remove_columns=dataset.column_names,
    )

    df = dataset.to_pandas()
    assert isinstance(df, DataFrame)
    df = df.rename(columns={"text": "corrected", "label": "level"})
    df.to_csv(output_path, index=False)
