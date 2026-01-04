Yoochoose full dat:
(torch_env) PS C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch> python run.py input_data\yoochoose-data\yoochoose_train_full.dat -ps loss=cross-entropy,layers=96,batch_size=128,dropout_p_embed=0.0,dropout_p_hidden=0.1,learning_rate=0.08,momentum=0.0,n_sample=128,sample_alpha=0.25,bpreg=0,logq=1,constrained_embedding=True,elu_param=0,n_epochs=5 -d cpu -s output_data\save_model_new.pt -t input_data\yoochoose-data\yoochoose_test.dat -m 20
d_embedding=True,elu_param=0,n_epochs=5 -d cpu -s output_data\x5csave_model_new.pt -t input_data\x5cyoochoose-data\x5cyoochoose_test.dat -m 20;7637b80d-243d-414d-b5c8-2664ffbcdd71Đang tạo mô hình GRU4Rec trên thiết bị "cpu"
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)  
ĐẶT   layers                  THÀNH   [96]            (kiểu: <class 'list'>) 
ĐẶT   batch_size              THÀNH   128             (kiểu: <class 'int'>)  
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.1             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.08            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   128             (kiểu: <class 'int'>)  
ĐẶT   sample_alpha            THÀNH   0.25            (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>) 
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_epochs                THÀNH   5               (kiểu: <class 'int'>)  
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data\yoochoose-data\yoochoose_train_full.dat   
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 75.59 giây
Đã tạo bộ lưu trữ mẫu với 78125 lô mẫu
C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\gru4rec_pytorch.py:567: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  self.P0 = torch.tensor(pop[self.data_iterator.itemidmap.index.values], dtype=torch.float32, device=self.device)
Epoch1 --> mất mát: 4.777750    (650.55s)       [230.94 mb/s | 29560 e/s]
Epoch2 --> mất mát: 4.565646    (689.74s)       [217.82 mb/s | 27881 e/s]
Epoch3 --> mất mát: 4.507774    (692.23s)       [217.03 mb/s | 27780 e/s]
Epoch4 --> mất mát: 4.478193    (676.10s)       [222.21 mb/s | 28443 e/s]
Epoch5 --> mất mát: 4.460087    (698.40s)       [215.12 mb/s | 27535 e/s]
Thời gian huấn luyện tổng cộng: 4317.08s
Đang lưu mô hình đã huấn luyện vào: output_data\save_model_new.pt
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data\yoochoose-data\yoochoose_test.dat
Bắt đầu đánh giá (cut-off=[20], sử dụng chế độ standard để xử lý hòa)
Original test data: 658146 events
Filtered test data: 608200 events (removed 49946 unknown items)
Training vocabulary size: 37800
Test data unique items: 19015
Items in both: 18854
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
Đánh giá mất 146.08s
Recall@20: 0.628131 MRR@20: 0.266718
Item coverage: 0.598651
Catalog coverage: 1.000000
ILD: 0.406557

Benchmark.py Yoochoose 5000 samples:
(torch_env) PS C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation>python web_demo\model\gru4rec_torch\benchmark.py yoochoose --model-path web_demo\model\gru4rec_torch\output_data\save_model_new.pt --test-samples 5000
Using preprocessed yoochoose-data dataset
[1/4] Loading data...
Loading data: C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\input_data\yoochoose-data\yoochoose_train_full.dat
Loading data: C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\input_data\yoochoose-data\yoochoose_test.dat
  Train: 27023131 events, 7802752 sessions
  Test:  658146 events, 179462 sessions
  Sampled test to 5000 sessions (18748 events)
[2/4] Loading GRU4Rec model...
  Loaded: web_demo\model\gru4rec_torch\output_data\save_model_new.pt
  Model vocabulary size: 37800
  Model layers: [96]
  Extracted embeddings shape: (37800, 96)
[3/4] Training baselines...
  MostPopular: 37800 items
  LastItem: ready (fallback to MostPopular if needed)
[4/4] Evaluating models...
  (Memory-efficient mode. Use --full-eval for complete diversity metrics)
  GRU4Rec...Original test data: 18748 events
Filtered test data: 17336 events (removed 1412 unknown items)
Training vocabulary size: 37800
Test data unique items: 4469
Items in both: 4362
Using existing item ID map
Dữ liệu đã được sắp xếp theo session_id, timestamp
 done
  MostPopular... done
  LastItem... done

================================================================================
BASELINE COMPARISON REPORT
================================================================================

RECALL and MRR Metrics:
--------------------------------------------------------------------------------
Model               Recall@20   MRR@20
--------------------------------------------------------------------------------
GRU4Rec             0.6225      0.2618
MostPopular         0.0056      0.0015
LastItem            0.3090      0.0974
--------------------------------------------------------------------------------

DIVERSITY Metrics (GRU4Rec):
--------------------------------------------------------------------------------
Intra-List Diversity: 0.4044

================================================================================

Retail Rocket full dat:
(torch_env) PS C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch> python run.py input_data\retailrocket-data\retailrocket_train_full.dat -ps loss=cross-entropy,layers=96,batch_size=128,dropout_p_embed=0.0,dropout_p_hidden=0.1,learning_rate=0.08,momentum=0.0,n_sample=128,sample_alpha=0.25,bpreg=0,logq=1,constrained_embedding=True,elu_param=0,n_epochs=5 -d cpu -s output_data\save_model_retailrocket.pt -t input_data\retailrocket-data\retailrocket_test.dat -m 20
trained_embedding=True,elu_param=0,n_epochs=5 -d cpu -s output_data\x5csave_model_retailrocket.pt -t input_data\x5cretailrocket-data\x5cretailrocket_test.dat -m 20;ba124add-a89c-4b0b-a867-5a1d03a0ac4bĐang tạo mô hình GRU4Rec trên thiết bị "cpu"
ĐẶT   loss                    THÀNH   cross-entropy   (kiểu: <class 'str'>)  
ĐẶT   layers                  THÀNH   [96]            (kiểu: <class 'list'>) 
ĐẶT   batch_size              THÀNH   128             (kiểu: <class 'int'>)  
ĐẶT   dropout_p_embed         THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   dropout_p_hidden        THÀNH   0.1             (kiểu: <class 'float'>)
ĐẶT   learning_rate           THÀNH   0.08            (kiểu: <class 'float'>)
ĐẶT   momentum                THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   n_sample                THÀNH   128             (kiểu: <class 'int'>)  
ĐẶT   sample_alpha            THÀNH   0.25            (kiểu: <class 'float'>)
ĐẶT   bpreg                   THÀNH   0.0             (kiểu: <class 'float'>)
ĐẶT   logq                    THÀNH   1.0             (kiểu: <class 'float'>)
ĐẶT   constrained_embedding   THÀNH   True            (kiểu: <class 'bool'>) 
ĐẶT   elu_param               THÀNH   0.0             (kiểu: <class 'float'>)   
ĐẶT   n_epochs                THÀNH   5               (kiểu: <class 'int'>)     
Đang tải dữ liệu huấn luyện...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data\retailrocket-data\retailrocket_train_full.dat
Bắt đầu huấn luyện
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 3.42 giây
Đã tạo bộ lưu trữ mẫu với 78125 lô mẫu
C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\gru4rec_pytorch.py:567: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  self.P0 = torch.tensor(pop[self.data_iterator.itemidmap.index.values], dtype=torch.float32, device=self.device)
Epoch1 --> mất mát: 3.872671    (36.22s)        [211.04 mb/s | 27013 e/s]
Epoch2 --> mất mát: 2.390993    (35.95s)        [212.65 mb/s | 27219 e/s]
Epoch3 --> mất mát: 1.907541    (35.49s)        [215.40 mb/s | 27572 e/s]
Epoch4 --> mất mát: 1.650709    (35.02s)        [218.28 mb/s | 27940 e/s]
Epoch5 --> mất mát: 1.487720    (36.95s)        [206.86 mb/s | 26478 e/s]
Thời gian huấn luyện tổng cộng: 218.51s
Đang lưu mô hình đã huấn luyện vào: output_data\save_model_retailrocket.pt      
Đang tải dữ liệu kiểm tra...
Đang tải dữ liệu từ tệp phân cách bằng TAB: input_data\retailrocket-data\retailrocket_test.dat
Bắt đầu đánh giá (cut-off=[20], sử dụng chế độ standard để xử lý hòa)
Original test data: 44910 events
Filtered test data: 44129 events (removed 781 unknown items)
Training vocabulary size: 85827
Test data unique items: 19777
Items in both: 19289
Using existing item ID map
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 0.09 giây
Đánh giá mất 24.86s
Recall@20: 0.394229 MRR@20: 0.121706
Item coverage: 0.408450
Catalog coverage: 1.000000
ILD: 0.461712

(torch_env) PS C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation>python web_demo\model\gru4rec_torch\benchmark.py retailrocket --model-path web_demo\model\gru4rec_torch\output_data\save_model_retailrocket.pt --test-samples 5000
Using preprocessed retailrocket-data dataset
[1/4] Loading data...
Loading data: C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\input_data\retailrocket-data\retailrocket_train_full.dat
Loading data: C:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch\input_data\retailrocket-data\retailrocket_test.dat
  Train: 1126705 events, 346186 sessions
  Test:  44910 events, 18026 sessions
  Sampled test to 5000 sessions (13154 events)
[2/4] Loading GRU4Rec model...
  Loaded: web_demo\model\gru4rec_torch\output_data\save_model_retailrocket.pt
  Model vocabulary size: 85827
  Model layers: [96]
  Extracted embeddings shape: (85827, 96)
[3/4] Training baselines...
  MostPopular: 85827 items
  LastItem: ready (fallback to MostPopular if needed)
[4/4] Evaluating models...
  (Memory-efficient mode. Use --full-eval for complete diversity metrics)
  GRU4Rec...Original test data: 13154 events
Filtered test data: 12932 events (removed 222 unknown items)
Training vocabulary size: 85827
Test data unique items: 8125
Items in both: 7961
Using existing item ID map
Dữ liệu chưa được sắp xếp theo session_id, đang sắp xếp...
Dữ liệu đã được sắp xếp trong 0.06 giây
 done
  MostPopular... done
  LastItem... done

================================================================================
BASELINE COMPARISON REPORT
================================================================================

RECALL and MRR Metrics:
--------------------------------------------------------------------------------
Model               Recall@20   MRR@20
--------------------------------------------------------------------------------
GRU4Rec             0.3712      0.1146
MostPopular         0.0048      0.0012
LastItem            0.1500      0.0559
--------------------------------------------------------------------------------

DIVERSITY Metrics (GRU4Rec):
--------------------------------------------------------------------------------
Intra-List Diversity: 0.4667

================================================================================

