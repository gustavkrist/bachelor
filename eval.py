import os
import pickle

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoModelForSequenceClassification, Trainer

from bachelor.dataset import get_dataset
from bachelor.metrics import fleisch_readability
from bachelor.utils import fk_to_cefr, level_to_cefr


def main() -> None:
    csv_path = f"{os.path.dirname(__file__)}/data/ef_POStagged_orig_corrected.csv"
    predictions_cache = f"{os.path.dirname(__file__)}/data/predictions.PKL"
    figure_path = f"{os.path.dirname(__file__)}/figures"
    dataset = get_dataset(csv_path)
    # dataset_torch = dataset.with_format("torch")
    # cefr_scores_fk = list(map(fk_to_cefr, map(fleisch_readability, dataset_torch["test"]["text"])))
    # if os.path.exists(predictions_cache):
    #     with open(predictions_cache, "rb") as f:
    #         cefr_predictions = pickle.load(f)
    # else:
    #     model = AutoModelForSequenceClassification.from_pretrained("trained_model", num_labels=16)
    #     trainer = Trainer(model=model)
    #     if torch.cuda.is_available():
    #         trainer.model = model.cuda()
    #     else:
    #         trainer.model = model
    #     predictions = trainer.predict(dataset_torch["test"])
    #     predicted_labels = predictions.predictions.argmax(axis=1)
    #     cefr_predictions = list(map(level_to_cefr, predicted_labels))
    #     with open(predictions_cache, "wb") as f:
    #         pickle.dump(cefr_predictions, f)
    # cefr_labels = list(map(lambda x: level_to_cefr(x.item()), dataset_torch["test"]["label"]))
    # print("FK accuracy:", accuracy_score(y_true=cefr_labels, y_pred=cefr_scores_fk))
    # print("Model accuracy:", accuracy_score(y_true=cefr_labels, y_pred=cefr_predictions))
    # os.makedirs(figure_path, exist_ok=True)
    # cmat = confusion_matrix(y_true=cefr_labels, y_pred=cefr_scores_fk, labels=["A1", "A2", "B1", "B2", "C1", "C2"])
    # ConfusionMatrixDisplay(cmat, display_labels=["A1", "A2", "B1", "B2", "C1", "C2"]).plot()
    # plt.title("FK confusion matrix on test data")
    # plt.savefig(f"{figure_path}/confusion_matrix-fk.png", bbox_inches="tight", dpi=400)
    # plt.clf()
    # cmat = confusion_matrix(y_true=cefr_labels, y_pred=cefr_predictions, labels=["A1", "A2", "B1", "B2", "C1", "C2"])
    # ConfusionMatrixDisplay(cmat, display_labels=["A1", "A2", "B1", "B2", "C1", "C2"]).plot()
    # plt.title("Model confusion matrix on test data")
    # plt.savefig(f"{figure_path}/confusion_matrix-model.png", bbox_inches="tight", dpi=400)
    # plt.clf()
    dataset_pandas = dataset.with_format("pandas")
    sns.histplot(
        pd.Categorical(
            list(map(level_to_cefr, dataset_pandas["train"]["label"])),
            ["A1", "A2", "B1", "B2", "C1", "C2"],
        ),
        stat="density",
    ).set(title="Train split distribution")
    plt.savefig(f"{figure_path}/train_distribution.png", bbox_inches="tight", dpi=400)
    plt.clf()
    sns.histplot(
        pd.Categorical(
            list(map(level_to_cefr, dataset_pandas["eval"]["label"])),
            ["A1", "A2", "B1", "B2", "C1", "C2"],
        ),
        stat="density",
    ).set(title="Eval split distribution")
    plt.savefig(f"{figure_path}/eval_distribution.png", bbox_inches="tight", dpi=400)
    plt.clf()
    sns.histplot(
        pd.Categorical(
            list(map(level_to_cefr, dataset_pandas["test"]["label"])),
            ["A1", "A2", "B1", "B2", "C1", "C2"],
        ),
        stat="density",
    ).set(title="Test split distribution")
    plt.savefig(f"{figure_path}/test_distribution.png", bbox_inches="tight", dpi=400)


if __name__ == "__main__":
    main()
