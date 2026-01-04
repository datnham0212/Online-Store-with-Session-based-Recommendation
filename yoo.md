LẦN 1 (YOOCHOOSE):
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0               (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [480]           (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   10              (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   48              (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.07            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048            (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 48.44 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
/kaggle/input/gru4rec-torch/gru4rec_torch/gru4rec_pytorch.py:567: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  self.P0 = torch.tensor(pop[self.data_iterator.itemidmap.index.values], dtype=torch.float32, device=self.device)
Epoch1 --> mất mát: 8.342652 	(1372.47s) 	[291.80 mb/s | 14007 e/s]
Epoch2 --> mất mát: 8.075387 	(1367.61s) 	[292.84 mb/s | 14056 e/s]
Epoch3 --> mất mát: 8.004959 	(1338.98s) 	[299.10 mb/s | 14357 e/s]
Epoch4 --> mất mát: 7.966856 	(1347.81s) 	[297.14 mb/s | 14263 e/s]
Epoch5 --> mất mát: 7.939441 	(1355.91s) 	[295.37 mb/s | 14178 e/s]
Epoch6 --> mất mát: 7.918726 	(1341.28s) 	[298.59 mb/s | 14332 e/s]
Epoch7 --> mất mát: 7.901504 	(1349.87s) 	[296.69 mb/s | 14241 e/s]
Epoch8 --> mất mát: 7.889631 	(1344.74s) 	[297.82 mb/s | 14295 e/s]
Epoch9 --> mất mát: 7.878832 	(1352.64s) 	[296.08 mb/s | 14212 e/s]
Epoch10 --> mất mát: 7.868377 	(1361.62s) 	[294.13 mb/s | 14118 e/s]
Thời gian huấn luyện tổng cộng: 13957.05s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/yoochoose_xe_winning_final.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 608200 events (removed 49946 unknown items)
Training vocabulary size: 37800
Test data unique items: 19015
Items in both: 18854
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Đánh giá mất 28.00s
Recall@1: 0.182009 MRR@1: 0.182009
Recall@5: 0.442042 MRR@5: 0.275749
Recall@10: 0.556175 MRR@10: 0.291174
Recall@20: 0.648169 MRR@20: 0.297629
Item coverage: 0.756217
Catalog coverage: 1.000000
ILD: 0.605261


TIẾP TỤC LẦN 1 (YOOCHOOSE):
(torch_env) PS C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch> python run.py output_data/yoochoose_xe_winning_final.pt -l -t input_data/yoochoose-data/yoochoose_test.dat -m 1 5 10 20 --eval-metrics diversity -d cpu
Đang tải mô hình đã huấn luyện từ tệp: output_data/yoochoose_xe_winning_final.pt (vào thiết bị "cpu")
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 608200 events (removed 49946 unknown items)
Training vocabulary size: 37800
Test data unique items: 19015
Items in both: 18854
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Building MinHash signatures: 100%|███████████████████████████████████████████████| 10000/10000 [00:20<00:00, 487.76it/s]
Querying LSH for candidate pairs: 100%|█████████████████████████████████████████| 10000/10000 [00:05<00:00, 1857.42it/s]
Đánh giá mất 261.98s
Aggregate diversity: 0.756217
Inter-user diversity: 0.888249



LẦN 2 (YOOCHOOSE):
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 96.1/96.1 kB 4.6 MB/s eta 0:00:00
/usr/local/lib/python3.12/dist-packages/sqlalchemy/orm/query.py:195: SyntaxWarning: "is not" with 'tuple' literal. Did you mean "!="?
  if entities is not ():
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0               (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [480]           (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   10              (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   48              (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.07            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048            (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 44.40 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Epoch1 --> mất mát: 8.344806 	(1451.49s) 	[275.92 mb/s | 13244 e/s]
Epoch2 --> mất mát: 8.076584 	(1440.25s) 	[278.07 mb/s | 13347 e/s]
Epoch3 --> mất mát: 8.006636 	(1436.23s) 	[278.85 mb/s | 13385 e/s]
Epoch4 --> mất mát: 7.967824 	(1442.94s) 	[277.55 mb/s | 13323 e/s]
Epoch5 --> mất mát: 7.938958 	(1433.88s) 	[279.31 mb/s | 13407 e/s]
Epoch6 --> mất mát: 7.920290 	(1459.51s) 	[274.40 mb/s | 13171 e/s]
Epoch7 --> mất mát: 7.902497 	(1440.37s) 	[278.05 mb/s | 13346 e/s]
Epoch8 --> mất mát: 7.889355 	(1440.58s) 	[278.01 mb/s | 13344 e/s]
Epoch9 --> mất mát: 7.879226 	(1439.00s) 	[278.31 mb/s | 13359 e/s]
Epoch10 --> mất mát: 7.869334 	(1435.34s) 	[279.02 mb/s | 13393 e/s]
Thời gian huấn luyện tổng cộng: 14827.51s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/yoochoose_xe_winning_final.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 608200 events (removed 49946 unknown items)
Training vocabulary size: 37800
Test data unique items: 19015
Items in both: 18854
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Building MinHash signatures: 100%|███████| 10000/10000 [00:11<00:00, 877.23it/s]
Querying LSH for candidate pairs: 100%|█| 10000/10000 [00:02<00:00, 4943.22it/s]
Đánh giá mất 51.55s
Recall@1: 0.181619 MRR@1: 0.181619
Recall@5: 0.441166 MRR@5: 0.275126
Recall@10: 0.556561 MRR@10: 0.290716
Recall@20: 0.648773 MRR@20: 0.297194
Item coverage: 0.754868
Catalog coverage: 1.000000
ILD: 0.602878
Aggregate diversity: 0.754868
Inter-user diversity: 0.884430


LẦN 1 (RETAIL ROCKET):
Đang tạo mô hình GRU4Rec trên thiết bị "cpu"
ĐẶT   loss                    THÀNH   bpr-max   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True      (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0         (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.5       (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [224]     (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   10        (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   80        (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.5       (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.05      (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.05      (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.4       (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048      (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.4       (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   1.95      (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   0.0       (kiểu: <class 'float'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 4.20 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Epoch1 --> mất mát: 0.490715    (223.77s)       [44.79 mb/s | 3583 e/s]
Epoch2 --> mất mát: 0.394413    (220.85s)       [45.38 mb/s | 3630 e/s]
Epoch3 --> mất mát: 0.357210    (218.44s)       [45.88 mb/s | 3670 e/s]
Epoch4 --> mất mát: 0.336137    (219.52s)       [45.65 mb/s | 3652 e/s]
Epoch5 --> mất mát: 0.322523    (214.66s)       [46.69 mb/s | 3735 e/s]
Epoch6 --> mất mát: 0.313393    (212.38s)       [47.19 mb/s | 3775 e/s]
Epoch7 --> mất mát: 0.306428    (211.11s)       [47.47 mb/s | 3798 e/s]
Epoch8 --> mất mát: 0.301348    (219.01s)       [45.76 mb/s | 3661 e/s]
Epoch9 --> mất mát: 0.297212    (225.14s)       [44.51 mb/s | 3561 e/s]
Epoch10 --> mất mát: 0.293761   (211.84s)       [47.31 mb/s | 3785 e/s]
Thời gian huấn luyện tổng cộng: 2222.74s
Đang lưu mô hình đã huấn luyện vào: output_data/retailrocket_bprmax_winning_final.pt        
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 44910 events
Filtered test data: 44129 events (removed 781 unknown items)
Training vocabulary size: 85827
Test data unique items: 19777
Items in both: 19289
Using existing item ID map
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 0.10 giây
Đánh giá mất 2421.17s
Recall@1: 0.118344 MRR@1: 0.118344
Recall@5: 0.283988 MRR@5: 0.176917
Recall@10: 0.370408 MRR@10: 0.188469
Recall@20: 0.458343 MRR@20: 0.194598
Item coverage: 0.549652
Catalog coverage: 1.000000
ILD: 0.611025
Aggregate diversity: 0.549652
Inter-user diversity: 0.998468

LẦN 2 (RETAIL ROCKET)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 96.1/96.1 kB 2.9 MB/s eta 0:00:00
/usr/local/lib/python3.12/dist-packages/sqlalchemy/orm/query.py:195: SyntaxWarning: "is not" with 'tuple' literal. Did you mean "!="?
  if entities is not ():
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
ĐẶT   loss                    THÀNH   bpr-max   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True      (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0         (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.5       (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [224]     (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   10        (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   80        (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.5       (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.05      (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.05      (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.4       (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048      (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.4       (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   1.95      (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   0.0       (kiểu: <class 'float'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 2.23 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Epoch1 --> mất mát: 0.490781 	(48.68s) 	[205.88 mb/s | 16470 e/s]
Epoch2 --> mất mát: 0.394515 	(48.28s) 	[207.56 mb/s | 16605 e/s]
Epoch3 --> mất mát: 0.357243 	(47.94s) 	[209.05 mb/s | 16724 e/s]
Epoch4 --> mất mát: 0.335954 	(48.11s) 	[208.33 mb/s | 16667 e/s]
Epoch5 --> mất mát: 0.322540 	(48.08s) 	[208.43 mb/s | 16674 e/s]
Epoch6 --> mất mát: 0.313186 	(47.97s) 	[208.91 mb/s | 16713 e/s]
Epoch7 --> mất mát: 0.306478 	(47.99s) 	[208.86 mb/s | 16708 e/s]
Epoch8 --> mất mát: 0.301164 	(47.87s) 	[209.35 mb/s | 16748 e/s]
Epoch9 --> mất mát: 0.297011 	(47.48s) 	[211.09 mb/s | 16888 e/s]
Epoch10 --> mất mát: 0.293610 	(48.02s) 	[208.72 mb/s | 16698 e/s]
Thời gian huấn luyện tổng cộng: 505.75s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/retailrocket_bprmax_winning_final.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 44910 events
Filtered test data: 44129 events (removed 781 unknown items)
Training vocabulary size: 85827
Test data unique items: 19777
Items in both: 19289
Using existing item ID map
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 0.05 giây
Building MinHash signatures: 100%|███████| 10000/10000 [00:13<00:00, 748.76it/s]
Querying LSH for candidate pairs: 100%|█| 10000/10000 [00:00<00:00, 18999.38it/s
Đánh giá mất 21.62s
Recall@1: 0.114822 MRR@1: 0.114822
Recall@5: 0.287965 MRR@5: 0.176225
Recall@10: 0.374801 MRR@10: 0.187780
Recall@20: 0.460312 MRR@20: 0.193749
Item coverage: 0.549198
Catalog coverage: 1.000000
ILD: 0.610797
Aggregate diversity: 0.549198
Inter-user diversity: 0.900107

