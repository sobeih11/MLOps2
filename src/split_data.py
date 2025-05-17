import os
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from src.logger import get_logger
logger = get_logger(__name__)

def split_and_save(cfg: DictConfig):
    cfg = cfg.pipeline.split

    df = pd.read_csv(cfg.data_path)
    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size)


    train_df = pd.concat([X_train, y_train.rename(cfg.target_col)], axis=1)
    test_df = pd.concat([X_test, y_test.rename(cfg.target_col)], axis=1)


    os.makedirs("data/processed", exist_ok=True)
    train_df.to_parquet("data/processed/train.parquet", index=False)
    test_df.to_parquet("data/processed/test.parquet", index=False)

    logger.info("Train/Test data saved in data/processed/")
