from pathlib import Path

import pandas as pd
import dvc.api
from sklearn.model_selection import train_test_split

from src.logger import get_logger

logger = get_logger(__name__)

def split_and_save(params):

    
    split_cfg = params["split"]
    data_path = Path(split_cfg["data_path"])
    target_col = split_cfg["target_col"]
    test_size = split_cfg["test_size"]
    seed = split_cfg.get("random_state", 42)


    df = pd.read_csv(data_path)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )


    train_df = pd.concat([X_train, y_train.rename(target_col)], axis=1)
    test_df = pd.concat([X_test, y_test.rename(target_col)], axis=1)

    processed_dir = Path("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    logger.info("Train/Test data saved in data/processed/")


if __name__ == "__main__":
    params = dvc.api.params_show()
    split_and_save(params)
