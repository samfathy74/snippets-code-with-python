# https://pypi.org/project/split-folders/
!pip install split-folders
import splitfolders

TRAIN_DATA_PATH="./train/"
VALID_DATA_PATH="./val"
RANDOM_SEED = 42
SPLIT_BY_MOVE = False
splitfolders.ratio(TRAIN_DATA_PATH, output=VALID_DATA_PATH, seed=RANDOM_SEED, ratio=(0.80, 0.20), move=SPLIT_BY_MOVE_OR_COPY)
