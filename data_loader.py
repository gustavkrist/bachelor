from argparse import ArgumentParser

import pandas as pd


def main() -> None:
    argparser = ArgumentParser()
    argparser.add_argument("path")
    argparser.add_argument("filetype", choices=("csv", "xml"))
    args = argparser.parse_args()


if __name__ == "__main__":
    main()
