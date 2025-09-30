from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss

@dataclass
class Stores:
    parquet_path: Path
    image_table_path: Path
    index_path: Path

class FeatureStore:
    def __init__(self, stores: Stores):
        self.stores = stores
        self.stores.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self.stores.image_table_path.parent.mkdir(parents=True, exist_ok=True)
        self.stores.index_path.parent.mkdir(parents=True, exist_ok=True)

    def append_features(self, df: pd.DataFrame):
        table = pa.Table.from_pandas(df)
        if self.stores.parquet_path.exists():
            pq.write_to_dataset(table, root_path=str(self.stores.parquet_path))
        else:
            pq.write_table(table, self.stores.parquet_path)

    def write_images(self, df: pd.DataFrame):
        pq.write_table(pa.Table.from_pandas(df), self.stores.image_table_path)

    def build_faiss(self, vecs: np.ndarray):
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vecs)
        index.add(vecs)
        faiss.write_index(index, str(self.stores.index_path))

    def read_all(self) -> pd.DataFrame:
        if self.stores.parquet_path.is_dir():
            return pq.ParquetDataset(str(self.stores.parquet_path)).read_pandas().to_pandas()
        return pd.read_parquet(self.stores.parquet_path)
