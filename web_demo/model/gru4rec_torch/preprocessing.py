import pandas as pd
import numpy as np
from datetime import timedelta
import glob
import os


def _ensure_datetime_series(s):
    """Return a pd.Series of datetimes from s (which may be int/float epoch seconds or strings)."""
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        # Treat as Unix epoch seconds
        return pd.to_datetime(s, unit='s', errors='coerce')
    # try to parse strings / mixed types
    return pd.to_datetime(s, errors='coerce')


def read_and_normalize(paths, dataset='yoochoose', use_events=None):
    def _read_file(path):
        if dataset == 'yoochoose':
            # Yoochoose-clicks.dat: SessionId, Timestamp, ItemId, Category
            # Auto-detect separator
            seps = ['\t', ',', ';', '|']
            df = None
            for sep in seps:
                try:
                    df = pd.read_csv(
                        path,
                        sep=sep,
                        header=None,
                        usecols=[0, 1, 2],
                        names=['SessionId', 'Timestamp', 'ItemId'],
                        dtype={'SessionId': object, 'ItemId': object},
                        low_memory=False
                    )
                    if df.shape[1] == 3:
                        break
                except Exception:
                    df = None
                    continue
            if df is None:
                raise ValueError(f"Cannot determine separator / read file {path}")
            # Normalize Timestamp: could be numeric epoch seconds or already date-like
            df['Timestamp'] = _ensure_datetime_series(df['Timestamp'])
        else:
            if path.lower().endswith('.xlsx'):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            df = df.rename(columns={
                'visitorid': 'SessionId',
                'itemid': 'ItemId',
                'event': 'Event',
                'timestamp': 'Timestamp'
            })
            if use_events and 'Event' in df.columns:
                df = df[df['Event'].isin(use_events)]
            df = df[['SessionId', 'ItemId', 'Timestamp']]
            df['Timestamp'] = _ensure_datetime_series(df['Timestamp'])
        # Drop rows with invalid timestamps or missing ids
        df = df.dropna(subset=['SessionId', 'ItemId', 'Timestamp'])
        return df

    if isinstance(paths, str) and any(c in paths for c in ['*', '?']):
        file_list = glob.glob(paths)
    elif isinstance(paths, (list, tuple)):
        file_list = list(paths)
    else:
        file_list = [paths]

    if not file_list:
        raise ValueError("No input files found for read_and_normalize")

    df_list = [_read_file(p) for p in file_list]
    df = pd.concat(df_list, ignore_index=True)
    # Ensure proper dtypes
    df['SessionId'] = df['SessionId'].astype(str)
    df['ItemId'] = df['ItemId'].astype(str)
    return df


def filter_data(df, min_session_length=2, min_item_support=5):
    # Lọc dữ liệu dựa trên độ dài session và số lần xuất hiện của item
    df = df.copy()
    item_counts = df['ItemId'].value_counts()
    valid_items = item_counts[item_counts >= min_item_support].index
    df = df[df['ItemId'].isin(valid_items)]

    sess_counts = df['SessionId'].value_counts()
    valid_sess = sess_counts[sess_counts >= min_session_length].index
    df = df[df['SessionId'].isin(valid_sess)]
    return df


def sort_and_dedup(df):
    # Sắp xếp và loại bỏ các mục trùng lặp trong cùng một session
    df = df.copy()
    # Ensure Timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = _ensure_datetime_series(df['Timestamp'])
    df = df.sort_values(['SessionId', 'Timestamp'])
    df['prev_item'] = df.groupby('SessionId')['ItemId'].shift(1)
    df = df[df['ItemId'] != df['prev_item']]
    return df.drop(columns=['prev_item'])


def map_indices(df, start_index=1):
    # Ánh xạ các item sang chỉ số duy nhất
    unique_items = pd.Index(df['ItemId'].unique())
    idx_map = {item: idx for idx, item in enumerate(unique_items, start=start_index)}
    df = df.copy()
    df['ItemIdx'] = df['ItemId'].map(idx_map)
    return df, idx_map


def split_time_based(df, test_days=7, valid_days=7):
    # Chia dữ liệu dựa trên thời gian
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = _ensure_datetime_series(df['Timestamp'])
    max_t = df['Timestamp'].max()
    if pd.isna(max_t):
        raise ValueError("Timestamp column contains no valid datetime values")
    test_boundary = max_t - timedelta(days=test_days)
    train_full = df[df['Timestamp'] < test_boundary].copy()
    test = df[df['Timestamp'] >= test_boundary].copy()

    max_train = train_full['Timestamp'].max()
    valid_boundary = max_train - timedelta(days=valid_days)
    train_tr = train_full[train_full['Timestamp'] < valid_boundary].copy()
    train_valid = train_full[train_full['Timestamp'] >= valid_boundary].copy()

    return {
        'train_full': train_full,
        'test': test,
        'train_tr': train_tr,
        'train_valid': train_valid
    }


def preprocess_pipeline(paths,
                        dataset,
                        use_events=None,
                        min_session_length=2,
                        min_item_support=5,
                        test_days=7,
                        valid_days=7,
                        index_start=1):
    # Pipeline xử lý dữ liệu
    df = read_and_normalize(paths, dataset, use_events)
    if df.empty:
        raise ValueError("No data returned from read_and_normalize")
    df = filter_data(df, min_session_length, min_item_support)
    if df.empty:
        raise ValueError("No data left after filter_data (adjust min_item_support / min_session_length)")
    df = sort_and_dedup(df)
    df, idx_map = map_indices(df, index_start)
    splits = split_time_based(df, test_days, valid_days)
    return splits, idx_map

if __name__ == '__main__':
    # Try sensible default locations but avoid hard failures if files absent.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # prefer local yoochoose-data folder if present, else fall back to older path
    possible_yoo = [
        os.path.join(repo_root, 'yoochoose-data', 'yoochoose-clicks.dat'),
        os.path.join(repo_root, 'yoochoose-clicks.dat'),
        os.path.join(repo_root, '..', 'yoochoose-data', 'yoochoose-clicks.dat')
    ]
    yoo_paths = next((p for p in possible_yoo if os.path.isfile(p)), None)
    if yoo_paths is None:
        print("Warning: Yoochoose file not found in expected locations. Set yoo_paths manually in __main__ if needed.")
    else:
        print("Using yoochoose file:", yoo_paths)
        yoo_splits, yoo_map = preprocess_pipeline(
            paths=yoo_paths,
            dataset='yoochoose',
            min_session_length=2,
            min_item_support=5,
            test_days=7,
            valid_days=7,
            index_start=1
        )
        output_dir = os.path.join(repo_root, 'output_data')
        os.makedirs(output_dir, exist_ok=True)
        for name, df in yoo_splits.items():
            # write Timestamp in ISO format for consistency
            out_df = df.copy()
            out_df.loc[:, 'Timestamp'] = out_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            out_df.to_csv(os.path.join(output_dir, f'yoochoose_{name}.dat'), index=False, sep='\t')
        pd.to_pickle(yoo_map, os.path.join(output_dir, 'yoochoose_map.pkl'))
        print("Yoochoose preprocessing complete. Files written to", output_dir)

    # Retail Rocket events.csv (only run if file present)
    possible_rr = [
        os.path.join(repo_root, 'retailrocket', 'events.csv'),
        os.path.join(repo_root, 'yoochoose-data', 'events.csv'),
        os.path.join(repo_root, '..', 'retailrocket', 'events.csv')
    ]
    rr_paths = next((p for p in possible_rr if os.path.isfile(p)), None)
    if rr_paths is None:
        print("RetailRocket events.csv not found in expected locations; skipping RetailRocket preprocessing.")
    else:
        print("Using RetailRocket file:", rr_paths)
        rr_splits, rr_map = preprocess_pipeline(
            paths=rr_paths,
            dataset='retailrocket',
            use_events=['view'],
            index_start=1
        )
        output_dir = os.path.join(repo_root, 'output_data')
        os.makedirs(output_dir, exist_ok=True)
        for name, df in rr_splits.items():
            out_df = df.copy()
            out_df.loc[:, 'Timestamp'] = out_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            out_df.to_csv(os.path.join(output_dir, f'retailrocket_{name}.dat'), index=False, sep='\t')
        pd.to_pickle(rr_map, os.path.join(output_dir, 'retailrocket_map.pkl'))
        print("RetailRocket preprocessing complete. Files written to", output_dir)
