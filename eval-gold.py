import os
import pickle

from collections import defaultdict
from datasets import Dataset
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoModelForSequenceClassification, Trainer

from bachelor.metrics import fleisch_readability
from bachelor.utils import fk_to_cefr, level_to_cefr
from bachelor.dataset import tokenize_dataset


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def get_dataset() -> Dataset:
    texts: dict[str, list[str]] = {"label": [], "text": []}
    for cefr in CEFR_LEVELS:
        path = f"{os.path.dirname(__file__)}/data/english/{cefr}"
        if os.path.exists(path):
            for filename in os.listdir(path):
                with open(f"{path}/{filename}") as f:
                    texts["label"].append(cefr)
                    texts["text"].append(f.read())
    return tokenize_dataset(Dataset.from_dict(texts))


def main() -> None:
    predictions_cache = f"{os.path.dirname(__file__)}/data/predictions-gold.PKL"
    dataset = get_dataset()
    dataset_torch = dataset.with_format("torch")
    dataset_torch = dataset_torch.map(lambda rows: {"label": [CEFR_LEVELS.index(x) for x in rows["label"]]}, batched=True)
    cefr_scores_fk = list(map(fk_to_cefr, map(fleisch_readability, dataset_torch["text"])))
    if os.path.exists(predictions_cache):
        with open(predictions_cache, "rb") as f:
            cefr_predictions = pickle.load(f)
    else:
        model = AutoModelForSequenceClassification.from_pretrained("trained_model", num_labels=16)
        trainer = Trainer(model=model)
        if torch.cuda.is_available():
            trainer.model = model.cuda()
        else:
            trainer.model = model
        predictions = trainer.predict(dataset_torch)
        predicted_labels = predictions.predictions.argmax(axis=1)
        cefr_predictions = list(map(level_to_cefr, predicted_labels))
        with open(predictions_cache, "wb") as f:
            pickle.dump(cefr_predictions, f)
    cefr_labels = dataset["label"]
    print("FK accuracy:", accuracy_score(y_true=cefr_labels, y_pred=cefr_scores_fk))
    print("Model accuracy:", accuracy_score(y_true=cefr_labels, y_pred=cefr_predictions))
    figure_path = f"{os.path.dirname(__file__)}/figures"
    os.makedirs(figure_path, exist_ok=True)
    cmat = confusion_matrix(y_true=cefr_labels, y_pred=cefr_scores_fk, labels=["A1", "A2", "B1", "B2", "C1", "C2"])
    ConfusionMatrixDisplay(cmat, display_labels=["A1", "A2", "B1", "B2", "C1", "C2"]).plot()
    plt.title("FK confusion matrix on test data")
    plt.savefig(f"{figure_path}/confusion_matrix-gold-fk.png", bbox_inches="tight", dpi=400)
    plt.cla()
    cmat = confusion_matrix(y_true=cefr_labels, y_pred=cefr_predictions, labels=["A1", "A2", "B1", "B2", "C1", "C2"])
    ConfusionMatrixDisplay(cmat, display_labels=["A1", "A2", "B1", "B2", "C1", "C2"]).plot()
    plt.title("Model confusion matrix on test data")
    plt.savefig(f"{figure_path}/confusion_matrix-gold-model.png", bbox_inches="tight", dpi=400)
    plt.cla()


if __name__ == "__main__":
    main()
