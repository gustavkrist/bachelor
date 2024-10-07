import os
from bachelor.dataset import get_dataset
from bachelor.model import train_model


def main() -> None:
    csv_path = f"{os.path.dirname(__file__)}/data/ef_POStagged_orig_corrected.csv"
    dataset = get_dataset(csv_path)
    train_model(dataset)


if __name__ == "__main__":
    main()
