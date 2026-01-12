# Aug 12
Goal: build a small Flask demo showing session-based recommendations using GRU4Rec (train small model or reuse Yoochoose checkpoint).
Early decision: create a tiny synthetic dataset for items a–f, train a CPU GRU4Rec model, and serve it via the web_demo.
Artifacts proposed/added:
generate_synth_af.py to create/train a tiny model (af_synth_model.pt) and mapping csv.
recommender.py wrapper to load model and expose recommend(session_items, topk).
Flask app (web_demo/app.py) integration that tracks session history, logs interactions, and shows top-k recommendations.
Run notes (Windows): commands to train in GRU4Rec repo and to start Flask app in web_demo; reminder to ensure PYTHONPATH or install package so gru4rec_pytorch is importable.

# Aug 13 updates: 
consolidated GRU4Rec code into gru4rec_torch, fixed sys.path import issues, added model inspection tooling (inspect_model.py), ensured model.eval() and error reporting in recommender.

# Aug 14 updates: 
switched to Yoochoose checkpoint (save_model_test.pt) for demo; patched imports and packaging; dynamic PRODUCTS generation; restored/enhanced interaction logging; added /log_click and /submit-checkout endpoints; recommender tested (catalog ~37.8k).

# Aug 15 updates: 
major preprocessing.py improvements:
robust timestamp parsing (_ensure_datetime_series), separator auto-detect, flexible path handling, dtype validation, safer CLI, returns (splits, idx_map).
Guidance: keep preprocessing offline (do not call per web request).

# Aug 17 updates: 
stabilized interaction logging (CSV header, timestamp, session UUID), per-client session UUID stored, fallback secret_key, UI fixes (topk=3, normalize IDs, _product_for_id stub), cluster plumbing skeleton added, recommendation fallback when history empty.
Current state: working session-based web demo that loads a pretrained GRU4Rec model, records richer interactions, and serves recommendations; preprocessing and clustering improvements are implemented but run offline.
Risks / next steps suggested:
Add session UUID persistently, popularity fallback, admin endpoint to retrain & reload, improve catalog size/metadata, finish clustering (use MiniBatchKMeans), and avoid running preprocessing on each request.

# Aug 24 updates:
Summary

Kept the large catalog in the recommender but avoided materializing it in the UI. Added paginated products API and client-side page loading. Fixed clickable recommendation items not resolving to product pages by caching lightweight stubs.
Work completed today

Server

Added paginated /products endpoint that accepts page & size and returns items + total.
Implemented _build_products(limit, offset) to slice recommender.itemidmap without building full list.
Made CATALOG_MAX determined lazily from recommender (len(itemidmap)) so "total" is available without materializing items.
Ensured interaction logging writes timestamp, session UUID, event type, item id, history length to data/interactions.csv.
Added logic in product_page to create and cache lightweight product stubs in global RECOMMENDATIONS so recommendation links resolve.
Loaded item cluster map on startup and exposed /clusters endpoint that groups PRODUCTS by cluster if available.
Kept offline preprocessing / clustering guidance: heavy ops are not run per-request.
Frontend

Added script (index.html) to fetch /products?page=N&size=M and append results to #product-grid (initial page + infinite scroll).
Logged visible product/recommendation IDs via /log_click to capture impressions.
Model / data

Recommender continues to load pretrained checkpoint model (model/gru4rec_torch/output_data/save_model_test.pt).
Catalog metadata remains lightweight stubs; full metadata can be backed by DB/disk in future.
Cluster CSV (yoochoose_item_clusters.csv) loaded if present to help candidate narrowing.
Verification / quick tests performed

curl /products?page=0&size=10 returns items + total.
Index page loads first page of products and infinite scroll triggers additional pages.
Clicking recommended item now resolves to a product page (stub created & cached) instead of 404 / not found.
Risks / follow-ups (next steps)

Add server-side caching for popular pages and rate-limit /products to avoid high load under concurrent requests.
Implement popularity-based fallback / precomputed candidate pools to avoid scoring entire catalog per request.
Export embeddings + add FAISS/ANN for fast candidate retrieval for large catalogs.
Store richer product metadata in SQLite/LMDB/RocksDB and fetch only the items requested by the UI.
Add admin endpoint to reload model/cluster maps and monitor recommender health.


# Aug 25 updates:
Summary — work completed today

Server

Added paginated /products API (page & size) returning items + total.
Implemented _build_products(limit, offset) to slice recommender.itemidmap without materializing the full catalog.
Made CATALOG_MAX determined lazily from recommender (len(itemidmap)) so the frontend can show total.
Stabilized interaction logging to data/interactions.csv (timestamp, session UUID, event type, item id, history length).
product_page now creates & caches lightweight product stubs in global RECOMMENDATIONS so recommendation links resolve.
Loaded item cluster map on startup and exposed /clusters endpoint (groups PRODUCTS by cluster if available).
Frontend

Added client-side pagination script in index.html: fetch /products?page=N&size=M, render pages, supports page-size selector and Prev/Next controls.
Implemented infinite-scroll fallback and impression logging of visible items via /log_click.
Removed per-card "View" button so cards are clickable directly; card layout updated to avoid duplicate ID display.
Model / data

Recommender still loads pretrained checkpoint (model/gru4rec_torch/output_data/save_model_test.pt).
Cluster CSV (yoochoose_item_clusters.csv) used to help candidate narrowing; catalog metadata kept as lightweight stubs.
Verification

curl /products?page=0&size=10 returns items + total.
Index loads first page and paging/infinite-scroll fetches more.
Clicking recommended items now resolves to product pages (stubs cached), avoiding 404s.
Risks / next steps (recommended)

Add server-side caching and rate-limiting for /products.
Implement popularity-based candidate pools or ANN (FAISS) for large-catalog retrieval.
Persist richer product metadata in a DB/KV store (SQLite/LMDB/RocksDB).
Add admin endpoint to reload model/cluster maps and monitor recommender health.


Làm sao biết được item nào để recommend. Dựa vào gì?

Chung:
Click vào A1 -> Kiếm trong dataset thì sẽ có khúc chứa danh sách những item tương tự: [A2, A3, A4]

Backend lấy ds item tương tự nó trả về.

Frontend hiển thị danh sách item tương tự ấy.

Chi tiết:
app.py có dòng: rec_ids = recommender.recommend(history, topk=6) 

Gọi def recommend(self, session_items, topk=6, exclude_seen=True) từ recommender.py

Trong recommender.py thì có:
class GRURecommender:
self.gru = GRU4Rec.loadmodel(model_path, device=device)

Tải model GRU4Rec rồi gán vào self.gru 

Quay lại recommender.py, def recommend(self, session_items, topk=6, exclude_seen=True)
làm công việc của nó rồi trả về return [str(self.idx_to_item[int(i)]) for i in top_indices]

Gán giá trị trả về ấy vào rec_ids bên app.py

Rồi từ app.py trả về kết quả ở phần frontend:
    # Fallback: if no recs (empty history), show first items as simple recommendations
    if not rec_ids and PRODUCTS:
        rec_ids = [p["id"] for p in PRODUCTS[:6]]
    # Build recommendation objects even if not in PRODUCTS
    recs = [_product_for_id(rid) for rid in rec_ids]      
    log_event("page_index", extra="|".join(history[-5:]))
    print(f"[INDEX] products={len(PRODUCTS)} history={history} rec_ids={rec_ids}")
    return render_template("index.html", products=PRODUCTS, recommendations=recs)


GRU4Rec làm công việc của nó xong, rồi nó trả về kết quả dưới dạng là topk
topk đưa qua frontend để hiển thị sản phẩm được recommend


Client-side clicks call the logging endpoint (see /log_click and the click handlers in index.html / web_demo/templates/product.html).
The server keeps a session history (see add_history / get_history); visiting a product page calls add_history(product_id) so the user’s recent item sequence is stored in session.
The index page builds recommendations by calling the model with that session history: index calls recommender.recommend(history, topk=6) (see web_demo/app.py).
The actual recommendation logic is in GRURecommender.recommend: it
maps item ids to model indices via itemidmap,
feeds the sequence into the GRU to produce a hidden state,
scores all items using the model output weights (Wy / By) via score_items,
optionally masks out already-seen items, and
returns the top‑k highest scoring item ids (mapped back with idx_to_item). See recommender.py.
Summary: recommendations are learned by the GRU4Rec model from training data and returned as the highest‑scoring items given the current session’s sequence of item ids.


# Aug 31 updates:
Đã thêm tóm tắt các lượt huấn luyện GRU4Rec hôm nay (các biến thể CPU siêu nhanh):

Các lần huấn luyện đã thực hiện:
1. sanity96_e2 (Kiểm tra nhanh nhỏ nhất)
   Lệnh:
   python run.py input_data/yoochoose-data/yoochoose_train_valid.dat -t input_data/yoochoose-data/yoochoose_test.dat -m 5 -ps loss=cross-entropy,layers=96,batch_size=128,dropout_p_embed=0.0,dropout_p_hidden=0.0,learning_rate=0.1,momentum=0.0,n_sample=128,sample_alpha=0.4,bpreg=0,logq=1,constrained_embedding=True,elu_param=0,n_epochs=2 -d cpu
   Mất mát theo epoch: [4.962238, 4.178009]
   Đánh giá (cutoff=5): Recall@5=0.389505  MRR@5=0.236573
   Tổng thời gian train ≈74.2s; đánh giá ≈54.8s.

2. gru4rec_ultra_fast_4u1 (Kiểu “full-train” rút gọn)
   Lệnh:
   python run.py input_data/yoochoose-data/yoochoose_train_valid.dat -t input_data/yoochoose-data/yoochoose_test.dat -m 5 10 20 -ps loss=cross-entropy,layers=128,batch_size=128,dropout_p_embed=0.0,dropout_p_hidden=0.15,learning_rate=0.08,momentum=0.0,n_sample=128,sample_alpha=0.25,bpreg=0,logq=1,constrained_embedding=True,elu_param=0,n_epochs=3 -d cpu -s output_data/gru4rec_ultra_fast_4u1.pt
   Mục đích: baseline nhanh đa cutoff. (Chưa ghi lại metric.)

3. gru4rec_ultra_fast_5u1 (Tập trung lặp nhanh)
   Lệnh:
   python run.py input_data/yoochoose-data/yoochoose_train_valid.dat -t input_data/yoochoose-data/yoochoose_test.dat -m 5 20 -ps loss=cross-entropy,layers=96,batch_size=192,dropout_p_embed=0.0,dropout_p_hidden=0.1,learning_rate=0.09,momentum=0.0,n_sample=64,sample_alpha=0.3,bpreg=0,logq=1,constrained_embedding=True,elu_param=0,n_epochs=5 -d cpu -s output_data/gru4rec_ultra_fast_5u1.pt
   Mục đích: vòng tinh chỉnh siêu nhanh.

4. gru4rec_ultra_fast_5u2 (Tăng nhẹ kích thước hidden)
   Lệnh:
   python run.py input_data/yoochoose-data/yoochoose_train_valid.dat -t input_data/yoochoose-data/yoochoose_test.dat -m 5 20 -ps loss=cross-entropy,layers=112,batch_size=192,dropout_p_embed=0.0,dropout_p_hidden=0.15,learning_rate=0.085,momentum=0.0,n_sample=96,sample_alpha=0.25,bpreg=0,logq=1,constrained_embedding=True,elu_param=0,n_epochs=6 -d cpu -s output_data/gru4rec_ultra_fast_5u2.pt
   Mục đích: so sánh tác động của tăng nhẹ layer + kích thước sampling.

Ghi chú:
- Tất cả chạy trên CPU; n_sample và kích thước layer giảm để tăng tốc.
- Tiếp theo: thu thập metric cho 4u1/5u1/5u2 (Recall@5/20, MRR) rồi quyết định cấu hình nào mở rộng.


# Sep 7 updates:
Xoá đi những đoạn code không cần thiết trên preprocessing.py
Đặt tên preprocessing.py lại thành yoochoose_preprocessing.py

# Sep 8 updates:
Sửa lại đường dẫn dataset trên máy Đạt
Xoá các output cũ, chạy lại 2 script tiền xử lý dataset yc & rr

# Sep 10 updates:
Giới hạn lại lượng items được log khi view sao cho chỉ log các items có render trên trang hiện tại
Ngừng gitignore interactions.csv 

# Sep 17 updates:
Thêm hàm mất mát TOP1 & TOP1-max
Thêm tiêu chí đánh giá Coverage & ILD (Intra-List Diversity) 

# Dec 12 updates:
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

# Dec 19 updates:
Removed interaction logging and model retraining functionality:
- Deleted web_demo/data/interactions.csv and data/ directory
- Deleted web_demo/retrain_gru.py (no longer retraining on logged interactions)
- Deleted web_demo/test_retrain_gru.py (test file for retrain module)
- Removed log_event() function and all interaction logging calls from app.py
- Removed /log_click endpoint (was used to log user interactions)
- Removed all references to INTERACTIONS_PATH and ENABLE_VIEW_LOGGING from app.py
- Removed interaction logging tests from test_app.py and test_e2e_integration.py
- Web demo now focuses purely on: loading pretrained model → serving recommendations → no interaction data collection

Rationale: Objectives do not include logging user interactions or retraining the model based on session data collected in the web demo. Model retraining is done offline with external datasets (Yoochoose, RetailRocket).

# Dec 28 updates:
First time:
=========================================
 RETAILROCKET - WINNING PARAMETERS TEST  
=========================================

Configuration:
  Dataset: RetailRocket (full)
  Loss: bpr-max
  Layers: 224
  Batch Size: 80
  Learning Rate: 0.05
  BPR Regularization: 1.95
  Epochs: 10
  Metrics: Recall@1, @5, @10, @20        

Training model...

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
Dữ liệu đã được sắp xếp trong 3.03 giây
Đã tạo bộ lưu trữ mẫu với 4882 lô mẫu
Epoch1 --> mất mát: 0.490686    (189.73s)       [52.82 mb/s | 4226 e/s]
Epoch2 --> mất mát: 0.394422    (192.85s)       [51.97 mb/s | 4157 e/s]
Epoch3 --> mất mát: 0.357239    (184.81s)       [54.23 mb/s | 4338 e/s]
Epoch4 --> mất mát: 0.336038    (186.81s)       [53.65 mb/s | 4292 e/s]
Epoch5 --> mất mát: 0.322525    (169.68s)       [59.06 mb/s | 4725 e/s]
Epoch6 --> mất mát: 0.313286    (170.15s)       [58.90 mb/s | 4712 e/s]
Epoch7 --> mất mát: 0.306358    (169.96s)       [58.97 mb/s | 4717 e/s]
Epoch8 --> mất mát: 0.301304    (178.29s)       [56.21 mb/s | 4497 e/s]
Epoch9 --> mất mát: 0.297041    (180.43s)       [55.55 mb/s | 4444 e/s]
Epoch10 --> mất mát: 0.293718   (194.79s)       [51.45 mb/s | 4116 e/s]
Thời gian huấn luyện tổng cộng: 1850.04s
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
Dữ liệu đã được sắp xếp trong 0.08 giây
Đánh giá mất 38.51s
Recall@1: 0.115693 MRR@1: 0.115693
Recall@5: 0.284102 MRR@5: 0.175604
Recall@10: 0.372680 MRR@10: 0.187390
Recall@20: 0.460009 MRR@20: 0.193455
Item coverage: 0.550701
Catalog coverage: 1.000000
ILD: 0.609921

=========================================
 TRAINING COMPLETE
=========================================

Second time:
=========================================
 RETAILROCKET - WINNING PARAMETERS TEST  
=========================================

Configuration:
  Dataset: RetailRocket (full)
  Loss: bpr-max
  Layers: 224
  Batch Size: 80
  Learning Rate: 0.05
  BPR Regularization: 1.95       
  Epochs: 10
  Metrics: Recall@1, @5, @10, @20

Training model...

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

=========================================
 TRAINING COMPLETE
=========================================
Results saved to: output_data/retailrocket_bprmax_winning_final.pt


# Jan 5th 2026
- GRU4Rec là mô hình từ năm 2016 (gần 10 năm trước), đã có nhiều cải tiến vượt trội như BERT4Rec, SASRec, SR-GNN.
- Nên thêm phần so sánh GRU4Rec với ít nhất 1-2 mô hình hiện đại hơn (Transformer-based)
Hoặc đề xuất cải tiến nhỏ cho GRU4Rec (ví dụ: attention mechanism, multi-behavior modeling)
<!-- - Cần thêm baseline như Item-KNN, FPMC, hoặc ít nhất là LSTM (Có KNN rồi) -->
<!-- - Cần làm rõ: mỗi phiên test có bao nhiêu bước dự đoán? Dùng toàn bộ phiên hay chỉ item cuối? -->
<!-- - Kết quả thực nghiệm chưa có độ lệch chuẩn (standard deviation). Không rõ các thí nghiệm được chạy bao nhiêu lần (random seeds) -->
<!-- - Thiếu các câu hỏi nghiên cứu nhé. Ví dụ:
  + RQ1: GRU4Rec đạt hiệu suất như thế nào trên bài toán SBRS?
  + RQ2: Ảnh hưởng của loss functions (CE vs BPR) như thế nào?
  + RQ3: Các yếu tố nào ảnh hưởng đến hiệu suất (layer size, batch, epochs)? -->

- Vài ý để làm cho ngon hơn:
Về việc cải tiến mô hình
Multi-behavior GRU4Rec:
Kết hợp click + add-to-cart + purchase (đã có trong Retail Rocket)
Gán trọng số khác nhau cho từng loại hành vi

Attention-enhanced GRU4Rec:
Thêm attention layer để học tầm quan trọng của từng item trong session
Đơn giản hơn Transformer nhưng hiệu quả hơn GRU thuần

Category-aware GRU4Rec:
Kết hợp thông tin category/metadata của sản phẩm
Giải quyết cold-start cho item mới


# Jan 9 2026: Giải thích cho implementation batching cũ 
PHÂN TÍCH KHO GTRÚC & LƯU TRỮ MÃ BATCHING
=========================================

Bạn đã phát hiện ra điều gì đó **rất quan trọng** - quan sát tuyệt vời!

**Triển khai thực tế nằm trong 4 tệp đó**, nhưng tôi đã bỏ qua chúng. Hãy phân tích những gì chúng thực sự làm:

1. **`batching_datasets.py`** - `SessionParallelDataset` 
   - Quản lý các phiên hoạt động và vị trí của chúng
   - Tạo `new_session_mask` để báo hiệu ranh giới phiên
   - **Đây là nơi cần kích hoạt việc đặt lại trạng thái ẩn**

2. **`batching_models.py`** - `SessionGRUModel`
   - Một **triển khai GRU độc lập** tách biệt từ GRU4Rec
   - Sử dụng `GRUCell` thay vì mô hình đầy đủ
   - **Có thể không khớp với kiến trúc GRU4Rec**

3. **`batching_demo.py`** - Tập lệnh kiểm thử
   - Chỉ kiểm thử với dữ liệu đồ chơi và phiên giả
   - Sử dụng `SessionGRUModel`, không phải GRU4Rec thực tế
   - **Không bao giờ được kiểm thử với GRU4Rec thực tế**

4. **`batching_utils.py`** - Trình tải
   - Cố gắng tải từ cột 'item_idx' (không tồn tại trong dữ liệu thực tế)
   - **Không tương thích với định dạng dữ liệu thực tế**

**Vấn đề thực tế:**

Cơ sở hạ tầng batching được **xây dựng riêng** và **không bao giờ được tích hợp với GRU4Rec thực tế**. Đó là một triển khai song song mà:
- Sử dụng `SessionGRUModel` của riêng nó thay vì `GRU4Rec`
- Mong đợi các định dạng dữ liệu khác nhau
- Có quản lý trạng thái ẩn của riêng nó
- Chỉ được kiểm thử trên dữ liệu đồ chơi

**Tại sao nó thất bại trong huấn luyện thực tế:**
- Mô hình `GRU4Rec` từ `gru4rec_pytorch.py` không tương thích với lớp batching này
- `BATCHING_IMPLEMENTATION_TEMPLATE.py` cố gắng buộc chúng cùng nhau, điều này phá vỡ mọi thứ

Đây là **mismatch kiến trúc cơ bản**, không chỉ các lỗi cần sửa. Batching thích hợp sẽ yêu cầu viết lại quá trình huấn luyện `GRU4Rec` để sử dụng tập dữ liệu batching một cách nhất quán.

Bạn muốn tôi ghi lại khoảng cách kiến trúc này và lưu trữ mã batching dưới dạng "nguyên mẫu nghiên cứu"?

Phát hiện: Triển khai batching song song phiên không tương thích về mặt kiến trúc với mô hình GRU4Rec sản xuất.

Phân tích Nguyên nhân Gốc:
- Cơ sở hạ tầng batching tồn tại trong 4 tệp riêng biệt:
  * batching_datasets.py - SessionParallelDataset (tạo new_session_mask)
  * batching_models.py - SessionGRUModel (triển khai GRU riêng, KHÔNG phải GRU4Rec)
  * batching_demo.py - Chỉ kiểm thử với dữ liệu tổng hợp, không bao giờ kiểm thử với GRU4Rec thực tế
  * batching_utils.py - Trình tải dữ liệu mong đợi cột 'item_idx' (dữ liệu thực tế có 'item_id')

- Các tệp này tạo thành một DỰ ÁN SONG SONG chưa bao giờ được tích hợp đúng cách:
  * Dự án A: gru4rec_pytorch.py + run.py + evaluation.py = HOẠT ĐỘNG (Recall@20=0.628)
  * Dự án B: thư mục batching/ = BỎ RƠI (SessionGRUModel tách biệt từ GRU4Rec)
  * Cầu nối thất bại: BATCHING_IMPLEMENTATION_TEMPLATE.py cố kết nối B→A = THẤT BẠI THẢM HỌC

Bằng chứng Thất bại Hiệu suất:
- Huấn luyện tiêu chuẩn: Recall@20=0.628, Loss=0.33, Time/epoch=226s
- Huấn luyện batching: Recall@20=0.016 (tệ hơn 97.5%), Loss=25.31 (cao hơn 76 lần), Time/epoch=1517s (chậm hơn 6.7 lần)
- Vấn đề gốc: SessionGRUModel chưa bao giờ được tích hợp với kiến trúc GRU4Rec thực tế
- Không tương thích quản lý trạng thái ẩn: lớp batching đặt lại trạng thái ẩn ở ranh giới phiên,
  nhưng GRU4Rec mong đợi huấn luyện liên tục với bảo toàn ngữ cảnh phiên thích hợp

Quyết định: Lưu trữ mã batching dưới dạng nguyên mẫu nghiên cứu (không xóa, nhưng đánh dấu là không hoạt động)
- Lý do: Cần viết lại kiến trúc đáng kể để batching hoạt động với GRU4Rec
- ROI: Tiềm năng tăng tốc biên so với nỗ lực cần thiết để tích hợp đầy đủ
- Phương án thay thế: Huấn luyện tiêu chuẩn với phương pháp đa-seed được chứng minh, có thể tái tạo được và không cần gỡ lỗi

Các Hành động Đã Thực hiện:
1. run_training_batching.py - Đã thêm cảnh báo không dùng nữa + lời nhắc chặn + so sánh hiệu suất
2. evaluation.py - Cố định hỗ trợ mô hình được tải (itemidmap, kiểm tra hasattr, nhập pandas)
3. Nhật ký này - Ghi lại khoảng cách kiến trúc để tham khảo trong tương lai

Khuyến cáo: Tiếp tục với quy trình huấn luyện tiêu chuẩn (run.py, run_multiseed.py).
Nếu cần tối ưu hóa batching trong tương lai:
- Tùy chọn 1: Viết lại hoàn toàn để sử dụng SessionGRUModel một cách nhất quán (không buộc vừa vào GRU4Rec)
- Tùy chọn 2: Triển khai lớp tích hợp thích hợp tôn trọng giả định session-liên tục của GRU4Rec
- Tùy chọn 3: Đánh giá các mô hình mới hơn (BERT4Rec, SASRec) có thể có hỗ trợ batching tốt hơn

Vị trí Lưu trữ: /batching/ (đánh dấu là nguyên mẫu nghiên cứu, không sẵn sàng sản xuất)