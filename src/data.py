import pandas as pd
import os
def load_data(relative_path: str) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    full_path = os.path.join(base_dir, relative_path)
    return pd.read_parquet(full_path)