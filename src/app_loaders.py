import json
import joblib
import pandas as pd

from pathlib import Path
# =========== AZURE STORAGE CONFIG ===========
import os
from io import BytesIO
from azure.storage.blob import BlobServiceClient

# ================= GLOBAL CACHE =================
model_cache = None
data_cache = None
region_mapping = None

# ================= REGION MAPPING =================
def load_region_mapping():
    global region_mapping
    if region_mapping is None:
        root_path = Path(__file__).parent.parent
        with open(root_path / "region_mapping.json", 'r') as f:
            region_mapping = json.load(f)

        # convert keys to int
        region_mapping = {int(k): v for k, v in region_mapping.items()}

    return region_mapping

# ================= LOAD MODEL =================
def load_model():
    global model_cache
    if model_cache is None:
        root_path = Path(__file__).parent.parent

        model_path = root_path / "models/xgb_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"❌ Model not found at {model_path}")

        model_cache = joblib.load(model_path)

        print("✅ Model loaded")

    return model_cache

# ================= LOAD DATA =================
def load_data():
    global data_cache

    if data_cache is None:
        # root_path = Path(__file__).parent.parent

        # # plot data (for map)
        # df_plot = pd.read_csv(root_path / "data/external/plot_data.csv")

        # # main dataset
        # df = pd.read_csv(root_path / "data/processed/final_data.csv")

        # ================= LOAD DATA FROM AZURE STORAGE =================
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

        if not connection_string:
            raise Exception("AZURE_STORAGE_CONNECTION_STRING not found")

        blob_service = BlobServiceClient.from_connection_string(connection_string)

        container = blob_service.get_container_client("data")

        plot_blob = container.download_blob("plot_data.csv")
        final_blob = container.download_blob("final_data.csv")

        df_plot = pd.read_csv(BytesIO(plot_blob.readall()))
        df = pd.read_csv(BytesIO(final_blob.readall()))

        # detect time column
        time_col = None
        for c in ["pickup_slot", "tpep_pickup_datetime", "pickup_datetime", "timestamp"]:
            if c in df.columns:
                time_col = c
                break

        if time_col is None:
            raise ValueError(f"❌ No time column found. Columns: {df.columns}")

        # convert datetime
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])

        # sort + index
        df = df.sort_values(time_col).set_index(time_col)

        print("✅ Data loaded")
        # print("Time column:", time_col)
        # print("Columns:", list(df.columns))

        data_cache = (df_plot, df)

    return data_cache