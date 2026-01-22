import os
os.chdir('/kaggle/input/g4r/pytorch/default/10/gru4rec_torch')

# Install GPU PyTorch
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install pandas numpy scikit-learn datasketch -q

# Fast experiment: Baseline GRU4Rec on Yoochoose (train_valid split) "web_demo\model\gru4rec_torch\paramfiles\yoochoose_xe_tuned_fast.py"
for seed in [42, 123, 456]:
    !python run.py input_data/yoochoose-data/yoochoose_train_valid.dat \
      -ps "loss=cross-entropy,constrained_embedding=True,embedding=0,elu_param=0.0,layers=112,n_epochs=4,batch_size=128,dropout_p_embed=0.0,dropout_p_hidden=0.1,learning_rate=0.08,momentum=0.0,n_sample=2048,sample_alpha=0.2,bpreg=0.0,logq=1.0" \
      -t input_data/yoochoose-data/yoochoose_test.dat \
      -m 1 5 10 20 \
      --eval-metrics recall_mrr,coverage,ild,diversity \
      -d cuda:0 \
      --seed {seed} \
      -s /kaggle/working/yoochoose_noat_fast_noat_seed{seed}.pt

# Fast experiment: Attention-GRU4Rec on Yoochoose (train_valid split)
for seed in [42, 123, 456]:
    !python run.py input_data/yoochoose-data/yoochoose_train_valid.dat \
      -ps "loss=cross-entropy,constrained_embedding=True,embedding=0,elu_param=0.0,layers=112,n_epochs=4,batch_size=128,dropout_p_embed=0.0,dropout_p_hidden=0.1,learning_rate=0.08,momentum=0.0,n_sample=2048,sample_alpha=0.2,bpreg=0.0,logq=1.0" \
      -t input_data/yoochoose-data/yoochoose_test.dat \
      -m 1 5 10 20 \
      --eval-metrics recall_mrr,coverage,ild,diversity \
      -d cuda:0 \
      --seed {seed} \
      -s /kaggle/working/yoochoose_fast_attn_seed{seed}.pt \
      --attention
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 96.5/96.5 kB 3.3 MB/s eta 0:00:00
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 42
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0               (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [112]           (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   4               (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   128             (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.1             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.08            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048            (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
ĐẶT   use_attention           THÀNH   False           (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_train_valid.dat
Bắt đầu huấn luyện
Dữ liệu đã được sắp xếp theo session_id, timestamp
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện
Epoch1 --> mất mát: 8.403824 	(21.81s) 	[272.28 mb/s | 34852 e/s]
Epoch2 --> mất mát: 7.689516 	(21.21s) 	[280.02 mb/s | 35842 e/s]
Epoch3 --> mất mát: 7.485318 	(21.20s) 	[280.17 mb/s | 35861 e/s]
Epoch4 --> mất mát: 7.363151 	(21.26s) 	[279.42 mb/s | 35765 e/s]
Thời gian huấn luyện tổng cộng: 102.63s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/yoochoose_noat_fast_noat_seed42.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 599803 events (removed 58343 unknown items)
Training vocabulary size: 20749
Test data unique items: 19015
Items in both: 16378
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Building MinHash signatures: 100%|███████| 10000/10000 [00:10<00:00, 913.37it/s]
Querying LSH for candidate pairs: 100%|██| 10000/10000 [00:43<00:00, 228.06it/s]
Đánh giá mất 80.65s
Recall@1: 0.173886 MRR@1: 0.173886
Recall@5: 0.418986 MRR@5: 0.261946
Recall@10: 0.531999 MRR@10: 0.277248
Recall@20: 0.626218 MRR@20: 0.283848
Item coverage: 0.785917
Catalog coverage: 1.000000
ILD: 0.511470
Aggregate diversity: 0.785917
Inter-user diversity: 0.877959
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 123
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0               (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [112]           (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   4               (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   128             (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.1             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.08            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048            (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
ĐẶT   use_attention           THÀNH   False           (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_train_valid.dat
Bắt đầu huấn luyện
Dữ liệu đã được sắp xếp theo session_id, timestamp
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện
Epoch1 --> mất mát: 8.412729 	(21.37s) 	[277.94 mb/s | 35576 e/s]
Epoch2 --> mất mát: 7.688206 	(21.02s) 	[282.47 mb/s | 36157 e/s]
Epoch3 --> mất mát: 7.480559 	(21.21s) 	[280.06 mb/s | 35848 e/s]
Epoch4 --> mất mát: 7.362474 	(21.25s) 	[279.51 mb/s | 35778 e/s]
Thời gian huấn luyện tổng cộng: 99.93s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/yoochoose_noat_fast_noat_seed123.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 599803 events (removed 58343 unknown items)
Training vocabulary size: 20749
Test data unique items: 19015
Items in both: 16378
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Building MinHash signatures: 100%|███████| 10000/10000 [00:11<00:00, 887.51it/s]
Querying LSH for candidate pairs: 100%|██| 10000/10000 [00:43<00:00, 231.85it/s]
Đánh giá mất 80.42s
Recall@1: 0.173740 MRR@1: 0.173740
Recall@5: 0.418970 MRR@5: 0.261925
Recall@10: 0.532100 MRR@10: 0.277201
Recall@20: 0.625879 MRR@20: 0.283780
Item coverage: 0.788809
Catalog coverage: 1.000000
ILD: 0.498136
Aggregate diversity: 0.788809
Inter-user diversity: 0.877304
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 456
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0               (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [112]           (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   4               (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   128             (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.1             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.08            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048            (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
ĐẶT   use_attention           THÀNH   False           (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_train_valid.dat
Bắt đầu huấn luyện
Dữ liệu đã được sắp xếp theo session_id, timestamp
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện
Epoch1 --> mất mát: 8.419677 	(21.52s) 	[275.95 mb/s | 35322 e/s]
Epoch2 --> mất mát: 7.693493 	(21.25s) 	[279.53 mb/s | 35780 e/s]
Epoch3 --> mất mát: 7.482811 	(21.17s) 	[280.52 mb/s | 35907 e/s]
Epoch4 --> mất mát: 7.364503 	(21.25s) 	[279.46 mb/s | 35771 e/s]
Thời gian huấn luyện tổng cộng: 100.42s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/yoochoose_noat_fast_noat_seed456.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 599803 events (removed 58343 unknown items)
Training vocabulary size: 20749
Test data unique items: 19015
Items in both: 16378
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Building MinHash signatures: 100%|███████| 10000/10000 [00:11<00:00, 903.99it/s]
Querying LSH for candidate pairs: 100%|██| 10000/10000 [00:45<00:00, 222.19it/s]
Đánh giá mất 82.52s
Recall@1: 0.173321 MRR@1: 0.173321
Recall@5: 0.418648 MRR@5: 0.261680
Recall@10: 0.532451 MRR@10: 0.277074
Recall@20: 0.626354 MRR@20: 0.283646
Item coverage: 0.789677
Catalog coverage: 1.000000
ILD: 0.505226
Aggregate diversity: 0.789677
Inter-user diversity: 0.874652
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 42
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0               (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [112]           (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   4               (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   128             (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.1             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.08            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048            (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
ĐẶT   use_attention           THÀNH   True            (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_train_valid.dat
Bắt đầu huấn luyện
Dữ liệu đã được sắp xếp theo session_id, timestamp
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện [Attention: dot, scale=0.1]
Epoch1 --> mất mát: 8.383879 	(22.80s) 	[260.53 mb/s | 33348 e/s]
Epoch2 --> mất mát: 7.661304 	(22.31s) 	[266.18 mb/s | 34071 e/s]
Epoch3 --> mất mát: 7.456402 	(22.10s) 	[268.76 mb/s | 34401 e/s]
Epoch4 --> mất mát: 7.335660 	(22.42s) 	[264.91 mb/s | 33908 e/s]
Thời gian huấn luyện tổng cộng: 104.83s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/yoochoose_fast_attn_seed42.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 599803 events (removed 58343 unknown items)
Training vocabulary size: 20749
Test data unique items: 19015
Items in both: 16378
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Building MinHash signatures: 100%|███████| 10000/10000 [00:11<00:00, 904.92it/s]
Querying LSH for candidate pairs: 100%|██| 10000/10000 [00:43<00:00, 229.72it/s]
Đánh giá mất 82.58s
Recall@1: 0.173594 MRR@1: 0.173594
Recall@5: 0.420689 MRR@5: 0.262403
Recall@10: 0.532698 MRR@10: 0.277540
Recall@20: 0.625734 MRR@20: 0.284056
Item coverage: 0.802593
Catalog coverage: 1.000000
ILD: 0.511611
Aggregate diversity: 0.802593
Inter-user diversity: 0.875640
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 123
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0               (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [112]           (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   4               (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   128             (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.1             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.08            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048            (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
ĐẶT   use_attention           THÀNH   True            (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_train_valid.dat
Bắt đầu huấn luyện
Dữ liệu đã được sắp xếp theo session_id, timestamp
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện [Attention: dot, scale=0.1]
Epoch1 --> mất mát: 8.395931 	(22.93s) 	[258.98 mb/s | 33149 e/s]
Epoch2 --> mất mát: 7.670276 	(22.67s) 	[261.97 mb/s | 33532 e/s]
Epoch3 --> mất mát: 7.461750 	(22.66s) 	[262.12 mb/s | 33552 e/s]
Epoch4 --> mất mát: 7.341389 	(22.81s) 	[260.41 mb/s | 33332 e/s]
Thời gian huấn luyện tổng cộng: 106.68s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/yoochoose_fast_attn_seed123.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 599803 events (removed 58343 unknown items)
Training vocabulary size: 20749
Test data unique items: 19015
Items in both: 16378
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Building MinHash signatures: 100%|███████| 10000/10000 [00:11<00:00, 873.70it/s]
Querying LSH for candidate pairs: 100%|██| 10000/10000 [00:43<00:00, 231.52it/s]
Đánh giá mất 83.45s
Recall@1: 0.174269 MRR@1: 0.174269
Recall@5: 0.420021 MRR@5: 0.262663
Recall@10: 0.532568 MRR@10: 0.277865
Recall@20: 0.625059 MRR@20: 0.284340
Item coverage: 0.802545
Catalog coverage: 1.000000
ILD: 0.512360
Aggregate diversity: 0.802545
Inter-user diversity: 0.879915
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 456
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>)
ĐẶT   embedding               THÀNH   0               (kiểu: <class 'int'>)
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   layers                  THÀNH   [112]           (kiểu: <class 'list'>)
ĐẶT   n_epochs                THÀNH   4               (kiểu: <class 'int'>)
ĐẶT   batch_size              THÀNH   128             (kiểu: <class 'int'>)
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.1             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.08            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   2048            (kiểu: <class 'int'>)
ĐẶT   sample_alpha            THÀNH   0.2             (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
ĐẶT   use_attention           THÀNH   True            (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_train_valid.dat
Bắt đầu huấn luyện
Dữ liệu đã được sắp xếp theo session_id, timestamp
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện [Attention: dot, scale=0.1]
Epoch1 --> mất mát: 8.391125 	(22.54s) 	[263.50 mb/s | 33728 e/s]
Epoch2 --> mất mát: 7.672928 	(22.14s) 	[268.23 mb/s | 34334 e/s]
Epoch3 --> mất mát: 7.463904 	(22.22s) 	[267.27 mb/s | 34211 e/s]
Epoch4 --> mất mát: 7.345248 	(22.25s) 	[266.91 mb/s | 34164 e/s]
Thời gian huấn luyện tổng cộng: 104.45s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/yoochoose_fast_attn_seed456.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/yoochoose-data/yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[1, 5, 10, 20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 599803 events (removed 58343 unknown items)
Training vocabulary size: 20749
Test data unique items: 19015
Items in both: 16378
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Building MinHash signatures: 100%|███████| 10000/10000 [00:10<00:00, 925.40it/s]
Querying LSH for candidate pairs: 100%|██| 10000/10000 [00:44<00:00, 224.76it/s]
Đánh giá mất 83.24s
Recall@1: 0.174107 MRR@1: 0.174107
Recall@5: 0.419692 MRR@5: 0.262529
Recall@10: 0.532091 MRR@10: 0.277714
Recall@20: 0.625007 MRR@20: 0.284223
Item coverage: 0.800761
Catalog coverage: 1.000000
ILD: 0.513463
Aggregate diversity: 0.800761
Inter-user diversity: 0.881218
import os
os.chdir('/kaggle/input/g4r/pytorch/default/10/gru4rec_torch')

# Install GPU PyTorch
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install pandas numpy scikit-learn datasketch -q

# Baseline GRU4Rec on RetailRocket (train_full, shared_best config)
for seed in [42, 123, 456]:
    !python run.py input_data/retailrocket-data/retailrocket_train_full.dat \
      -ps "loss=bpr-max,constrained_embedding=True,embedding=0,elu_param=0.5,layers=224,n_epochs=10,batch_size=80,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0" \
      -t input_data/retailrocket-data/retailrocket_test.dat \
      -m 1 5 10 20 \
      --eval-metrics recall_mrr,coverage,ild,diversity \
      -d cuda:0 \
      --seed {seed} \
      -s /kaggle/working/retailrocket_best_seed{seed}.pt

# # Attention-GRU4Rec on RetailRocket (train_full, shared_best config)
# for seed in [42, 123, 456]:
#     !python run.py input_data/retailrocket-data/retailrocket_train_full.dat \
#       -ps "loss=bpr-max,constrained_embedding=True,embedding=0,elu_param=0.5,layers=224,n_epochs=10,batch_size=80,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0" \
#       -t input_data/retailrocket-data/retailrocket_test.dat \
#       -m 1 5 10 20 \
#       --eval-metrics recall_mrr,coverage,ild,diversity \
#       -d cuda:0 \
#       --seed {seed} \
#       -s /kaggle/working/retailrocket_best_attn_seed{seed}.pt \
#       --attention --attention_scale 0.5
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 96.5/96.5 kB 3.1 MB/s eta 0:00:00
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 42
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
ĐẶT   use_attention           THÀNH   False     (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 2.54 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện
Epoch1 --> mất mát: 0.490275 	(48.29s) 	[207.52 mb/s | 16602 e/s]
Epoch2 --> mất mát: 0.394675 	(44.85s) 	[223.44 mb/s | 17875 e/s]
Epoch3 --> mất mát: 0.357628 	(48.07s) 	[208.49 mb/s | 16679 e/s]
Epoch4 --> mất mát: 0.336506 	(48.51s) 	[206.61 mb/s | 16529 e/s]
Epoch5 --> mất mát: 0.322751 	(48.03s) 	[208.64 mb/s | 16691 e/s]
Epoch6 --> mất mát: 0.313633 	(48.70s) 	[205.81 mb/s | 16465 e/s]
Epoch7 --> mất mát: 0.306653 	(49.32s) 	[203.20 mb/s | 16256 e/s]
Epoch8 --> mất mát: 0.301413 	(48.59s) 	[206.25 mb/s | 16500 e/s]
Epoch9 --> mất mát: 0.297176 	(48.28s) 	[207.58 mb/s | 16607 e/s]
Epoch10 --> mất mát: 0.293679 	(48.44s) 	[206.91 mb/s | 16553 e/s]
Thời gian huấn luyện tổng cộng: 507.06s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/retailrocket_best_seed42.pt
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
Building MinHash signatures: 100%|███████| 10000/10000 [00:13<00:00, 767.84it/s]
Querying LSH for candidate pairs: 100%|█| 10000/10000 [00:06<00:00, 1443.09it/s]
Đánh giá mất 30.11s
Recall@1: 0.117019 MRR@1: 0.117019
Recall@5: 0.286299 MRR@5: 0.177470
Recall@10: 0.371809 MRR@10: 0.188981
Recall@20: 0.459744 MRR@20: 0.195090
Item coverage: 0.550922
Catalog coverage: 1.000000
ILD: 0.611085
Aggregate diversity: 0.550922
Inter-user diversity: 0.898825
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 123
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
ĐẶT   use_attention           THÀNH   False     (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 2.53 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện
Epoch1 --> mất mát: 0.490695 	(49.80s) 	[201.23 mb/s | 16099 e/s]
Epoch2 --> mất mát: 0.394531 	(48.80s) 	[205.37 mb/s | 16430 e/s]
Epoch3 --> mất mát: 0.357259 	(46.68s) 	[214.69 mb/s | 17175 e/s]
Epoch4 --> mất mát: 0.336252 	(48.83s) 	[205.23 mb/s | 16419 e/s]
Epoch5 --> mất mát: 0.322664 	(49.18s) 	[203.80 mb/s | 16304 e/s]
Epoch6 --> mất mát: 0.313393 	(49.14s) 	[203.95 mb/s | 16316 e/s]
Epoch7 --> mất mát: 0.306515 	(49.34s) 	[203.11 mb/s | 16249 e/s]
Epoch8 --> mất mát: 0.301439 	(49.19s) 	[203.75 mb/s | 16300 e/s]
Epoch9 --> mất mát: 0.297135 	(49.37s) 	[202.99 mb/s | 16239 e/s]
Epoch10 --> mất mát: 0.293836 	(49.59s) 	[202.08 mb/s | 16167 e/s]
Thời gian huấn luyện tổng cộng: 513.03s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/retailrocket_best_seed123.pt
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
Building MinHash signatures: 100%|███████| 10000/10000 [00:12<00:00, 780.45it/s]
Querying LSH for candidate pairs: 100%|█| 10000/10000 [00:06<00:00, 1478.69it/s]
Đánh giá mất 29.75s
Recall@1: 0.115656 MRR@1: 0.115656
Recall@5: 0.283951 MRR@5: 0.175440
Recall@10: 0.369727 MRR@10: 0.186928
Recall@20: 0.460842 MRR@20: 0.193226
Item coverage: 0.550573
Catalog coverage: 1.000000
ILD: 0.609640
Aggregate diversity: 0.550573
Inter-user diversity: 0.896933
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 456
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
ĐẶT   use_attention           THÀNH   False     (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 2.47 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện
Epoch1 --> mất mát: 0.490209 	(49.37s) 	[203.01 mb/s | 16241 e/s]
Epoch2 --> mất mát: 0.394062 	(48.95s) 	[204.74 mb/s | 16379 e/s]
Epoch3 --> mất mát: 0.357114 	(47.76s) 	[209.83 mb/s | 16786 e/s]
Epoch4 --> mất mát: 0.335884 	(46.40s) 	[215.98 mb/s | 17278 e/s]
Epoch5 --> mất mát: 0.322480 	(48.74s) 	[205.64 mb/s | 16451 e/s]
Epoch6 --> mất mát: 0.313169 	(49.58s) 	[202.12 mb/s | 16169 e/s]
Epoch7 --> mất mát: 0.306494 	(48.84s) 	[205.19 mb/s | 16415 e/s]
Epoch8 --> mất mát: 0.301265 	(49.05s) 	[204.34 mb/s | 16347 e/s]
Epoch9 --> mất mát: 0.297019 	(49.00s) 	[204.51 mb/s | 16361 e/s]
Epoch10 --> mất mát: 0.293422 	(49.15s) 	[203.91 mb/s | 16313 e/s]
Thời gian huấn luyện tổng cộng: 509.86s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/retailrocket_best_seed456.pt
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
Building MinHash signatures: 100%|███████| 10000/10000 [00:12<00:00, 787.21it/s]
Querying LSH for candidate pairs: 100%|█| 10000/10000 [00:07<00:00, 1416.38it/s]
Đánh giá mất 29.97s
Recall@1: 0.115504 MRR@1: 0.115504
Recall@5: 0.284860 MRR@5: 0.175614
Recall@10: 0.372188 MRR@10: 0.187216
Recall@20: 0.462660 MRR@20: 0.193512
Item coverage: 0.548545
Catalog coverage: 1.000000
ILD: 0.611957
Aggregate diversity: 0.548545
Inter-user diversity: 0.898563
usage: run.py [-h] [-ps PARAM_STRING] [-pf PARAM_PATH] [-l] [-s MODEL_PATH]
              [-t TEST_PATH [TEST_PATH ...]] [-m AT [AT ...]] [-e EVAL_TYPE]
              [-ss SS] [-g GRFILE] [-d D] [-ik IK] [-sk SK] [-tk TK]
              [-pm METRIC] [-lpm] [--eval-metrics EVAL_METRICS] [--seed SEED]
              [--attention]
              PATH
run.py: error: unrecognized arguments: --attention_scale 0.5
usage: run.py [-h] [-ps PARAM_STRING] [-pf PARAM_PATH] [-l] [-s MODEL_PATH]
              [-t TEST_PATH [TEST_PATH ...]] [-m AT [AT ...]] [-e EVAL_TYPE]
              [-ss SS] [-g GRFILE] [-d D] [-ik IK] [-sk SK] [-tk TK]
              [-pm METRIC] [-lpm] [--eval-metrics EVAL_METRICS] [--seed SEED]
              [--attention]
              PATH
run.py: error: unrecognized arguments: --attention_scale 0.5
usage: run.py [-h] [-ps PARAM_STRING] [-pf PARAM_PATH] [-l] [-s MODEL_PATH]
              [-t TEST_PATH [TEST_PATH ...]] [-m AT [AT ...]] [-e EVAL_TYPE]
              [-ss SS] [-g GRFILE] [-d D] [-ik IK] [-sk SK] [-tk TK]
              [-pm METRIC] [-lpm] [--eval-metrics EVAL_METRICS] [--seed SEED]
              [--attention]
              PATH
run.py: error: unrecognized arguments: --attention_scale 0.5
import os
os.chdir('/kaggle/input/g4r/pytorch/default/10/gru4rec_torch')

# Install GPU PyTorch
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install pandas numpy scikit-learn datasketch -q

# Attention-GRU4Rec on RetailRocket (train_full, shared_best config)
for seed in [42, 123, 456]:
    !python run.py input_data/retailrocket-data/retailrocket_train_full.dat \
      -ps "loss=bpr-max,constrained_embedding=True,embedding=0,elu_param=0.5,layers=224,n_epochs=10,batch_size=80,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0" \
      -t input_data/retailrocket-data/retailrocket_test.dat \
      -m 1 5 10 20 \
      --eval-metrics recall_mrr,coverage,ild,diversity \
      -d cuda:0 \
      --seed {seed} \
      -s /kaggle/working/retailrocket_best_attn_seed{seed}.pt \
      --attention
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 96.5/96.5 kB 3.3 MB/s eta 0:00:00
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 42
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
ĐẶT   use_attention           THÀNH   True      (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 2.44 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện [Attention: dot, scale=0.1]
Epoch1 --> mất mát: 0.490388 	(50.45s) 	[198.67 mb/s | 15893 e/s]
Epoch2 --> mất mát: 0.393994 	(49.58s) 	[202.12 mb/s | 16170 e/s]
Epoch3 --> mất mát: 0.356617 	(49.49s) 	[202.51 mb/s | 16201 e/s]
Epoch4 --> mất mát: 0.335381 	(49.67s) 	[201.76 mb/s | 16141 e/s]
Epoch5 --> mất mát: 0.321900 	(49.53s) 	[202.33 mb/s | 16186 e/s]
Epoch6 --> mất mát: 0.312584 	(49.46s) 	[202.63 mb/s | 16211 e/s]
Epoch7 --> mất mát: 0.305934 	(49.71s) 	[201.61 mb/s | 16129 e/s]
Epoch8 --> mất mát: 0.300792 	(50.04s) 	[200.29 mb/s | 16023 e/s]
Epoch9 --> mất mát: 0.296632 	(49.99s) 	[200.48 mb/s | 16038 e/s]
Epoch10 --> mất mát: 0.293155 	(49.53s) 	[202.35 mb/s | 16188 e/s]
Thời gian huấn luyện tổng cộng: 523.28s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/retailrocket_best_attn_seed42.pt
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
Building MinHash signatures: 100%|███████| 10000/10000 [00:13<00:00, 745.62it/s]
Querying LSH for candidate pairs: 100%|█| 10000/10000 [00:07<00:00, 1396.05it/s]
Đánh giá mất 32.00s
Recall@1: 0.114595 MRR@1: 0.114595
Recall@5: 0.285087 MRR@5: 0.175563
Recall@10: 0.369802 MRR@10: 0.186881
Recall@20: 0.458191 MRR@20: 0.193008
Item coverage: 0.552518
Catalog coverage: 1.000000
ILD: 0.614394
Aggregate diversity: 0.552518
Inter-user diversity: 0.901779
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 123
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
ĐẶT   use_attention           THÀNH   True      (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 2.36 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện [Attention: dot, scale=0.1]
Epoch1 --> mất mát: 0.490624 	(48.61s) 	[206.15 mb/s | 16492 e/s]
Epoch2 --> mất mát: 0.394399 	(48.36s) 	[207.23 mb/s | 16578 e/s]
Epoch3 --> mất mát: 0.357077 	(48.69s) 	[205.85 mb/s | 16468 e/s]
Epoch4 --> mất mát: 0.335954 	(48.20s) 	[207.91 mb/s | 16633 e/s]
Epoch5 --> mất mát: 0.322269 	(48.62s) 	[206.12 mb/s | 16490 e/s]
Epoch6 --> mất mát: 0.313039 	(48.16s) 	[208.11 mb/s | 16649 e/s]
Epoch7 --> mất mát: 0.306198 	(48.15s) 	[208.16 mb/s | 16653 e/s]
Epoch8 --> mất mát: 0.300979 	(48.96s) 	[204.72 mb/s | 16377 e/s]
Epoch9 --> mất mát: 0.296796 	(49.57s) 	[202.18 mb/s | 16175 e/s]
Epoch10 --> mất mát: 0.293416 	(49.62s) 	[201.98 mb/s | 16158 e/s]
Thời gian huấn luyện tổng cộng: 509.65s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/retailrocket_best_attn_seed123.pt
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
Building MinHash signatures: 100%|███████| 10000/10000 [00:13<00:00, 757.26it/s]
Querying LSH for candidate pairs: 100%|█| 10000/10000 [00:07<00:00, 1411.32it/s]
Đánh giá mất 31.77s
Recall@1: 0.116375 MRR@1: 0.116375
Recall@5: 0.286185 MRR@5: 0.177021
Recall@10: 0.373817 MRR@10: 0.188711
Recall@20: 0.463304 MRR@20: 0.194918
Item coverage: 0.550619
Catalog coverage: 1.000000
ILD: 0.611496
Aggregate diversity: 0.550619
Inter-user diversity: 0.897467
Đang tạo mô hình GRU4Rec trên thiết bị "cuda:0"
Random seed set to: 456
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
ĐẶT   use_attention           THÀNH   True      (kiểu: <class 'bool'>)
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 2.55 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Bắt đầu huấn luyện [Attention: dot, scale=0.1]
Epoch1 --> mất mát: 0.490110 	(49.01s) 	[204.47 mb/s | 16358 e/s]
Epoch2 --> mất mát: 0.393768 	(48.62s) 	[206.14 mb/s | 16491 e/s]
Epoch3 --> mất mát: 0.356585 	(48.55s) 	[206.44 mb/s | 16515 e/s]
Epoch4 --> mất mát: 0.335448 	(48.59s) 	[206.25 mb/s | 16500 e/s]
Epoch5 --> mất mát: 0.321930 	(48.58s) 	[206.29 mb/s | 16503 e/s]
Epoch6 --> mất mát: 0.312780 	(48.21s) 	[207.86 mb/s | 16629 e/s]
Epoch7 --> mất mát: 0.306128 	(48.54s) 	[206.48 mb/s | 16519 e/s]
Epoch8 --> mất mát: 0.300795 	(47.97s) 	[208.94 mb/s | 16715 e/s]
Epoch9 --> mất mát: 0.296703 	(48.15s) 	[208.12 mb/s | 16650 e/s]
Epoch10 --> mất mát: 0.293237 	(48.28s) 	[207.60 mb/s | 16608 e/s]
Thời gian huấn luyện tổng cộng: 508.97s
Đang lưu mô hình đã huấn luyện vào: /kaggle/working/retailrocket_best_attn_seed456.pt
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
Building MinHash signatures: 100%|███████| 10000/10000 [00:12<00:00, 779.39it/s]
Querying LSH for candidate pairs: 100%|█| 10000/10000 [00:06<00:00, 1570.49it/s]
Đánh giá mất 30.83s
Recall@1: 0.116943 MRR@1: 0.116943
Recall@5: 0.285579 MRR@5: 0.177173
Recall@10: 0.372226 MRR@10: 0.188736
Recall@20: 0.462092 MRR@20: 0.194996
Item coverage: 0.551645
Catalog coverage: 1.000000
ILD: 0.611134
Aggregate diversity: 0.551645
Inter-user diversity: 0.897714