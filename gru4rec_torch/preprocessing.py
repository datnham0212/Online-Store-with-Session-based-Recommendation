import pandas as pd
import numpy as np
from datetime import timedelta
import glob
import os


def read_and_normalize(paths, dataset='yoochoose', use_events=None):
    def _read_file(path):
        if dataset == 'yoochoose':
            # Yoochoose-clicks.dat: SessionId, Timestamp, ItemId, Category
            # Tự động dò ký tự phân tách
            seps = ['\t', ',', ';', '|']
            for sep in seps:
                try:
                    df = pd.read_csv(
                        path,
                        sep=sep,
                        header=None,
                        usecols=[0, 1, 2],
                        names=['SessionId', 'Timestamp', 'ItemId'],  # Cập nhật tên cột
                        parse_dates=['Timestamp']
                    )
                    # Nếu đúng 3 cột thì trả về luôn
                    if df.shape[1] == 3:
                        break
                except Exception:
                    continue
            else:
                raise ValueError(f"Không thể xác định ký tự phân tách phù hợp cho file {path}")

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
            if use_events:
                df = df[df['Event'].isin(use_events)]
            df = df[['SessionId', 'ItemId', 'Timestamp']]
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df

    if isinstance(paths, str) and any(c in paths for c in ['*', '?']):
        file_list = glob.glob(paths)
    elif isinstance(paths, (list, tuple)):
        file_list = list(paths)
    else:
        file_list = [paths]

    df_list = [_read_file(p) for p in file_list]
    df = pd.concat(df_list, ignore_index=True)
    return df


def filter_data(df, min_session_length=2, min_item_support=5):
    # Lọc dữ liệu dựa trên độ dài session và số lần xuất hiện của item
    item_counts = df['ItemId'].value_counts()
    valid_items = item_counts[item_counts >= min_item_support].index
    df = df[df['ItemId'].isin(valid_items)]

    sess_counts = df['SessionId'].value_counts()
    valid_sess = sess_counts[sess_counts >= min_session_length].index
    df = df[df['SessionId'].isin(valid_sess)]
    return df


def sort_and_dedup(df):
    # Sắp xếp và loại bỏ các mục trùng lặp trong cùng một session
    df = df.sort_values(['SessionId', 'Timestamp'])
    df['prev_item'] = df.groupby('SessionId')['ItemId'].shift(1)
    df = df[df['ItemId'] != df['prev_item']]
    return df.drop(columns=['prev_item'])


def map_indices(df, start_index=1):
    # Ánh xạ các item sang chỉ số duy nhất
    unique_items = df['ItemId'].unique()
    idx_map = {item: idx for idx, item in enumerate(unique_items, start=start_index)}
    df['ItemIdx'] = df['ItemId'].map(idx_map)
    return df, idx_map


def split_time_based(df, test_days=7, valid_days=7):
    # Chia dữ liệu dựa trên thời gian
    max_t = df['Timestamp'].max()
    test_boundary = max_t - timedelta(days=test_days)
    train_full = df[df['Timestamp'] < test_boundary]
    test = df[df['Timestamp'] >= test_boundary]

    max_train = train_full['Timestamp'].max()
    valid_boundary = max_train - timedelta(days=valid_days)
    train_tr = train_full[train_full['Timestamp'] < valid_boundary]
    train_valid = train_full[train_full['Timestamp'] >= valid_boundary]

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
    df = filter_data(df, min_session_length, min_item_support)
    df = sort_and_dedup(df)
    df, idx_map = map_indices(df, index_start)
    splits = split_time_based(df, test_days, valid_days)
    return splits, idx_map

if __name__ == '__main__':
    # Yoochoose-clicks
    yoo_paths = r'c:\Users\Admin\Documents\Research\GRU4Rec_PyTorch_Official\yoochoose-data\yoochoose-clicks.dat'
    yoo_splits, yoo_map = preprocess_pipeline(
        paths=yoo_paths,
        dataset='yoochoose',
        min_session_length=2,
        min_item_support=5,
        test_days=7,
        valid_days=7,
        index_start=1
    )
    output_dir = r'c:\Users\Admin\Documents\Research\GRU4Rec_PyTorch_Official\output_data'
    os.makedirs(output_dir, exist_ok=True)
    for name, df in yoo_splits.items():
        df.to_csv(os.path.join(output_dir, f'yoochoose_{name}.dat'), index=False, sep='\t')
    pd.to_pickle(yoo_map, os.path.join(output_dir, 'yoochoose_map.pkl'))

    # Retail Rocket events.csv
    rr_paths = r'c:\Users\Admin\Documents\Research\GRU4Rec_PyTorch_Official\yoochoose-data\events.csv'
    rr_splits, rr_map = preprocess_pipeline(
        paths=rr_paths,
        dataset='retailrocket',
        use_events=['view']
    )
    for name, df in rr_splits.items():
        df.to_csv(os.path.join(output_dir, f'retailrocket_{name}.dat'), index=False, sep='\t')
    pd.to_pickle(rr_map, os.path.join(output_dir, 'retailrocket_map.pkl'))