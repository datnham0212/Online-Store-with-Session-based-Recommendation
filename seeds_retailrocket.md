(torch_env) PS C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch> python run_multiseed.py retailrocket-data 3

╔════════════════════════════════════════════════════════════════╗
║  MULTI-SEED EVALUATION FOR REPRODUCIBILITY                   ║
╚════════════════════════════════════════════════════════════════╝

Dataset:        retailrocket-data
Number of runs: 3
Seeds:          [42, 123, 456]
Parameter file: paramfiles/retailrocket_bprmax_shared_best.py

Configuration Parameters:
loss=bpr-max,constrained_embedding=True,embedding=0,elu_param=0.5,layers=224,n_epochs=10,batch_size=80,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0


================================================================================
RUN 42: Training with seed=42
================================================================================

Đang tạo mô hình GRU4Rec trên thiết bị "cpu"
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
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 5.44 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Epoch1 --> mất mát: 0.490985    (278.45s)       [35.99 mb/s | 2879 e/s]
Epoch2 --> mất mát: 0.394916    (275.03s)       [36.44 mb/s | 2915 e/s]
Epoch3 --> mất mát: 0.357510    (272.03s)       [36.84 mb/s | 2947 e/s]
Epoch4 --> mất mát: 0.336178    (256.20s)       [39.12 mb/s | 3129 e/s]
Epoch5 --> mất mát: 0.322654    (271.48s)       [36.92 mb/s | 2953 e/s]
Epoch6 --> mất mát: 0.313273    (288.62s)       [34.72 mb/s | 2778 e/s]
Epoch7 --> mất mát: 0.306513    (260.45s)       [38.48 mb/s | 3078 e/s]
Epoch8 --> mất mát: 0.301185    (240.21s)       [41.72 mb/s | 3338 e/s]
Epoch9 --> mất mát: 0.296970    (234.06s)       [42.82 mb/s | 3425 e/s]
Epoch10 --> mất mát: 0.293604   (212.61s)       [47.14 mb/s | 3771 e/s]
Thời gian huấn luyện tổng cộng: 2657.24s
Đang lưu mô hình đã huấn luyện vào: output_data/best_retailrocket_seed42.pt
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
Dữ liệu đã được sắp xếp trong 0.13 giây
Đánh giá mất 34.38s
Recall@1: 0.116299 MRR@1: 0.116299
Recall@5: 0.284367 MRR@5: 0.176540
Recall@10: 0.371999 MRR@10: 0.188287
Recall@20: 0.460501 MRR@20: 0.194436
Item coverage: 0.550444
Catalog coverage: 1.000000
ILD: 0.610904

================================================================================
RUN 123: Training with seed=123
================================================================================

Đang tạo mô hình GRU4Rec trên thiết bị "cpu"
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
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 3.34 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Epoch1 --> mất mát: 0.490370    (232.29s)       [43.14 mb/s | 3452 e/s]
Epoch2 --> mất mát: 0.394370    (223.99s)       [44.74 mb/s | 3579 e/s]
Epoch3 --> mất mát: 0.357285    (211.44s)       [47.40 mb/s | 3792 e/s]
Epoch4 --> mất mát: 0.336169    (321.22s)       [31.20 mb/s | 2496 e/s]
Epoch5 --> mất mát: 0.322455    (309.86s)       [32.34 mb/s | 2588 e/s]
Epoch6 --> mất mát: 0.313297    (275.15s)       [36.42 mb/s | 2914 e/s]
Epoch7 --> mất mát: 0.306440    (275.36s)       [36.40 mb/s | 2912 e/s]
Epoch8 --> mất mát: 0.301227    (277.69s)       [36.09 mb/s | 2887 e/s]
Epoch9 --> mất mát: 0.297067    (277.74s)       [36.08 mb/s | 2887 e/s]
Epoch10 --> mất mát: 0.293675   (280.73s)       [35.70 mb/s | 2856 e/s]
Thời gian huấn luyện tổng cộng: 2724.82s
Đang lưu mô hình đã huấn luyện vào: output_data/best_retailrocket_seed123.pt
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
Dữ liệu đã được sắp xếp trong 0.09 giây
Đánh giá mất 41.98s
Recall@1: 0.115996 MRR@1: 0.115996
Recall@5: 0.286033 MRR@5: 0.176606
Recall@10: 0.374763 MRR@10: 0.188457
Recall@20: 0.460804 MRR@20: 0.194392
Item coverage: 0.551155
Catalog coverage: 1.000000
ILD: 0.610428

================================================================================
RUN 456: Training with seed=456
================================================================================

Đang tạo mô hình GRU4Rec trên thiết bị "cpu"
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
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data/retailrocket-data/retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 4.65 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Epoch1 --> mất mát: 0.491599    (273.32s)       [36.67 mb/s | 2933 e/s]
Epoch2 --> mất mát: 0.395325    (270.64s)       [37.03 mb/s | 2962 e/s]
Epoch3 --> mất mát: 0.357991    (285.58s)       [35.09 mb/s | 2807 e/s]
Epoch4 --> mất mát: 0.336443    (277.65s)       [36.10 mb/s | 2888 e/s]
Epoch5 --> mất mát: 0.322784    (278.42s)       [36.00 mb/s | 2880 e/s]
Epoch6 --> mất mát: 0.313610    (282.08s)       [35.53 mb/s | 2842 e/s]
Epoch7 --> mất mát: 0.306712    (327.19s)       [30.63 mb/s | 2450 e/s]
Epoch8 --> mất mát: 0.301447    (292.26s)       [34.29 mb/s | 2743 e/s]
Epoch9 --> mất mát: 0.297147    (325.91s)       [30.75 mb/s | 2460 e/s]
Epoch10 --> mất mát: 0.293760   (376.62s)       [26.61 mb/s | 2129 e/s]
Thời gian huấn luyện tổng cộng: 3039.65s
Đang lưu mô hình đã huấn luyện vào: output_data/best_retailrocket_seed456.pt
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
Dữ liệu đã được sắp xếp trong 0.14 giây
Đánh giá mất 47.97s
Recall@1: 0.116830 MRR@1: 0.116830
Recall@5: 0.282777 MRR@5: 0.175785
Recall@10: 0.370370 MRR@10: 0.187481
Recall@20: 0.460009 MRR@20: 0.193771
Item coverage: 0.547671
Catalog coverage: 1.000000
ILD: 0.608338

================================================================================
SUMMARY
================================================================================
Successful runs: 3/3
  Seed 42: ✓ PASS
  Seed 123: ✓ PASS
  Seed 456: ✓ PASS

Models saved to:
  output_data/best_retailrocket_seed42.pt
  output_data/best_retailrocket_seed123.pt
  output_data/best_retailrocket_seed456.pt

Next step: Extract metrics from output above
          python -c "import numpy as np; recalls=[...]; print(f'Recall@20: {np.mean(recalls):.4f} ± {np.std(recalls, ddof=1):.4f}')"

===========================================================================
RETAILROCKET - BPR-Max (3 seeds)
===========================================================================

📊 RANKING METRICS (Recall & MRR):
---------------------------------------------------------------------------
RECALL_1        0.116375 ± 0.000412
RECALL_5        0.284392 ± 0.001539
RECALL_10       0.372377 ± 0.002065
RECALL_20       0.460438 ± 0.000379

MRR_1           0.116375 ± 0.000412
MRR_5           0.176477 ± 0.000404
MRR_10          0.188075 ± 0.000481
MRR_20          0.194200 ± 0.000303

🎯 COVERAGE & DIVERSITY METRICS:
---------------------------------------------------------------------------
Item Coverage   0.549757 ± 0.001633
Catalog Coverage 1.000000 ± 0.000000
ILD             0.609890 ± 0.001200
AGGREGATE_DIVERSITY      0.550177 ± 0.000525
INTER_USER_DIVERSITY     0.949288 ± 0.049180


================================================================================
SUMMARY TABLE (n=5 runs - Dec 28 + Seeds 42, 123, 456)
================================================================================
Metric                    Mean         Std          CV (%)
--------------------------------------------------------------------------------
Recall@1                  0.116375 ± 0.000412      0.35%
Recall@5                  0.284392 ± 0.001539      0.54%
Recall@10                 0.372377 ± 0.002065      0.55%
Recall@20                 0.460438 ± 0.000379      0.08%
MRR@1                     0.116375 ± 0.000412      0.35%
MRR@5                     0.176477 ± 0.000404      0.23%
MRR@10                    0.188075 ± 0.000481      0.26%
MRR@20                    0.194200 ± 0.000303      0.16%
Item Coverage             0.549757 ± 0.001633      0.30%
Catalog Coverage          1.000000 ± 0.000000      0.00%
ILD                       0.609890 ± 0.001200      0.20%
Aggregate Diversity       0.550177 ± 0.000525      0.10%
Inter-User Diversity      0.949288 ± 0.049180      5.18%