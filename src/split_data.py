import os
import pandas as pd
import yaml
import dvc.api
from sklearn.model_selection import train_test_split
from logger import get_logger

logger = get_logger(__name__)

def split_and_save(params):

    
    split_cfg = params["split"]
    data_path = split_cfg["data_path"]
    target_col = split_cfg["target_col"]
    test_size = split_cfg["test_size"]


    df = pd.read_csv(data_path)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)


    train_df = pd.concat([X_train, y_train.rename(target_col)], axis=1)
    test_df = pd.concat([X_test, y_test.rename(target_col)], axis=1)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_parquet("data/processed/train.parquet", index=False)
    test_df.to_parquet("data/processed/test.parquet", index=False)

    logger.info("Train/Test data saved in data/processed/")


if __name__ == "__main__":
    params = dvc.api.params_show()
    split_and_save(params)
