from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Value,
)
from transformers import AutoTokenizer


def load_dataset(
    path: str,
    features: Features | None = None,
    cols: list[str] | None = None,
    train_size: float = 0.8,
) -> DatasetDict:
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
    dataset = dataset.rename_columns({"level": "label", "corrected": "text"}).shuffle(seed=0)
    train_test_splits = dataset.train_test_split(train_size=train_size, seed=0)
    evaluate_test_splits = train_test_splits["test"].train_test_split(train_size=0.5, seed=0)
    return DatasetDict(
        {
            "train": train_test_splits["train"],
            "eval": evaluate_test_splits["train"],
            "test": evaluate_test_splits["test"],
        }
    )


def filter_dataset(dataset: DatasetDict, minimum_grade: int = 80) -> DatasetDict:
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


def get_dataset(path: str) -> DatasetDict:
    return tokenize_dataset(filter_dataset(load_dataset(path)))
