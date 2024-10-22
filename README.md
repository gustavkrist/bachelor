# Gustav's Bachelor Project

This repo includes scripts and a library for training a model for predicting the difficulty of
a provided text. The difficulty is classified in 16 labels from the [EFCAMDAT corpus](https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html).
The training dataset is also from the EFCAMDAT corpus, namely the file `ef_POStagged_orig_corrected.csv`.
You must yourself obtain access to the corpus and download this file if you wish to train the
model. The specified file must be placed in a directory called `data` at the repo root.

The model can be trained by running `python3 run.py` and will be output in the `trained_model`
directory.
