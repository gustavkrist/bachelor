from typing import cast
import os

import evaluate
import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


def train_model(dataset: DatasetDict) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=16
    )
    training_args = TrainingArguments(output_dir="model_checkpoints", eval_strategy="epoch")
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, int]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return cast(dict[str, int], metric.compute(predictions=predictions, references=labels))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    output_dir = os.path.dirname(os.path.dirname(__file__))
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(output_dir=output_dir)
