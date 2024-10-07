import os
from sklearn.metrics import accuracy_score

import torch
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer
)

from bachelor.dataset import get_dataset
from bachelor.metrics import fleisch_readability
from bachelor.utils import fk_to_cefr, level_to_cefr


def main() -> None:
    csv_path = f"{os.path.dirname(__file__)}/data/ef_POStagged_orig_corrected.csv"
    dataset = get_dataset(csv_path).with_format("torch")
    cefr_scores_fk = list(
        map(fk_to_cefr, map(fleisch_readability, dataset["test"]["text"]))
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "trained_model", num_labels=16
    )
    trainer = Trainer(model=model)
    trainer.model = model.cuda()
    predictions = trainer.predict(dataset["test"])
    predicted_labels = predictions.predictions.argmax(axis=1)
    cefr_predictions = list(map(level_to_cefr, predicted_labels))
    cefr_labels = list(map(lambda x: level_to_cefr(x.item()), dataset["test"]["label"]))
    print("FK accuracy:", accuracy_score(y_true=cefr_labels, y_pred=cefr_scores_fk))
    print("Model accuracy:", accuracy_score(y_true=cefr_labels, y_pred=cefr_predictions))


if __name__ == "__main__":
    main()
