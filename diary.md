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