# So Sánh GRU4Rec với Các Baseline Đơn Giản (MostPopular & LastItem)

## 1. Tại Sao Cần Baseline?

Baseline là các mô hình **đơn giản, không có tham số học** dùng để:
- Chứng minh GRU4Rec thực sự học được thứ gì từ dữ liệu
- Kiểm tra xem GRU4Rec có thực sự tốt hơn các phương pháp naive hay không
- Xác định performance floor: điểm cơ sở mà bất kỳ mô hình tối thiểu cũng phải vượt qua

---

## 2. Hai Baseline Được Sử Dụng

### MostPopularBaseline
**Ý tưởng:** Luôn gợi ý các item **phổ biến nhất toàn cục** (được click nhiều nhất trong training)

**Cách hoạt động:**
```python
# Bước 1: Đếm tần suất item trong dữ liệu huấn luyện
item_counts = {item_1: 1000, item_2: 800, item_3: 500, ...}

# Bước 2: Sắp xếp theo tần suất giảm dần
popular_items = [item_1, item_2, item_3, ...]

# Bước 3: Gợi ý top-K item phổ biến nhất
recommend(session_items, topk=20) → [item_1, item_2, ..., item_20]
```

**Đặc điểm:**
- [+] Không có tham số học (không cần training)
- [+] Rất nhanh (O(k) tìm top-K)
- [-] **Hoàn toàn bỏ qua lịch sử phiên** → không cá nhân hóa
- [-] Không học mối quan hệ giữa items
- [-] Khuynh hướng gợi ý những item luôn phổ biến → thiếu đa dạng

### LastItemBaseline
**Ý tưởng:** Gợi ý các item **tương tự với item cuối cùng** mà người dùng đã click

**Cách hoạt động:**
```python
# Bước 1: Lấy item cuối cùng
last_item = session_items[-1]

# Bước 2: Lấy embedding của item cuối
last_emb = item_embeddings[last_item]

# Bước 3: Tính cosine similarity với tất cả item
for item in all_items:
    emb = item_embeddings[item]
    similarity[item] = cos(last_emb, emb) = (last_emb · emb) / (||last_emb|| · ||emb||)

# Bước 4: Sắp xếp theo similarity giảm dần
recommend(topk=20) → top-20 item có similarity cao nhất
```

**Đặc điểm:**
- [+] Sử dụng 1 item gần đây (tính chất "nearness")
- [+] Dùng embedding để tính similarity (có chút học tập)
- [-] **Chỉ nhìn item cuối cùng, bỏ qua phần còn lại** của phiên
- [-] Không học temporal pattern (thứ tự click)
- [-] Không phát hiện được các mối quan hệ phức tạp

**Fallback:** Nếu item cuối không có embedding → fallback sang MostPopular

---

## 3. Kết Quả So Sánh Thực Tế

### Yoochoose Dataset (Toàn Bộ Training Set)
**Setup:** Train GRU4Rec trên 27M events, test trên 658K events, cutoff=20

| Model | Recall@20 | MRR@20 | Coverage |
|-------|-----------|--------|----------|
| **GRU4Rec** | **0.6281** | **0.2667** | **0.5987** |
| LastItem | 0.3090 | 0.0974 | - |
| MostPopular | 0.0056 | 0.0015 | - |

**Nhận xét:**
- GRU4Rec vượt trội **2×** so với LastItem (Recall 0.628 vs 0.309)
- GRU4Rec vượt trội **112×** so với MostPopular (Recall 0.628 vs 0.0056)
- **LastItem khá tốt**, chứng tỏ "gợi ý dựa trên item cuối" có logic, nhưng vẫn kém GRU4Rec đáng kể
- **MostPopular rất tệ**, xác nhận rằng **bỏ qua lịch sử phiên là sai lầm**

---

### RetailRocket Dataset (Toàn Bộ Training Set)
**Setup:** Train GRU4Rec trên 1.1M events, test trên 44.9K events, cutoff=20

| Model | Recall@20 | MRR@20 | Coverage |
|-------|-----------|--------|----------|
| **GRU4Rec** | **0.3942** | **0.1217** | **0.4085** |
| LastItem | 0.1500 | 0.0559 | - |
| MostPopular | 0.0048 | 0.0012 | - |

**Nhận xét:**
- GRU4Rec vượt trội **2.6×** so với LastItem (Recall 0.394 vs 0.150)
- GRU4Rec vượt trội **82×** so với MostPopular (Recall 0.394 vs 0.0048)
- Mẫu nhất quán: GRU4Rec > LastItem >> MostPopular
- RetailRocket nhỏ hơn Yoochoose → recall thấp hơn, nhưng pattern vẫn đúng

---

## 4. Phân Tích Sâu Hơn

### 4.1 Tại Sao GRU4Rec Vượt Trội LastItem?

**LastItem chỉ nhìn item cuối cùng:**
```
Phiên: [laptop → mouse → charger → (target: cable)]

LastItem:
  - Lấy embedding của "charger"
  - Tìm item tương tự "charger"
  - Có thể gợi ý: adapter, cord, charger_cable, ...
  - [-] Bỏ qua việc người dùng mới mua laptop → có thể cần phần mềm, bảo hành
  - [-] Bỏ qua thứ tự temporal: laptop → accessories

GRU4Rec:
  - Xử lý cả chuỗi: [laptop → mouse → charger]
  - Mô hình Gated Recurrent Unit **update hidden state** qua từng bước
  - hidden_state_1 = GRU(laptop_emb, hidden_0)
  - hidden_state_2 = GRU(mouse_emb, hidden_state_1)
  - hidden_state_3 = GRU(charger_emb, hidden_state_2) → predict "cable"
  - [+] Nhận biết được "đã mua laptop → cần accessories" (long-range dependency)
  - [+] Học thứ tự temporal và progression pattern
```

### 4.2 Tại Sao MostPopular Tệ Đến Vậy?

**MostPopular gợi ý những item phổ biến nhất → không cá nhân hóa:**
```
Training data stats:
  Item A: 50000 lần click (phổ biến nhất)
  Item B: 40000 lần click
  Item C: 30000 lần click
  ...

MostPopular sẽ luôn gợi ý: [A, B, C, D, E, ...]
cho TẤT CẢ người dùng, bất kể phiên họ là gì.

Ví dụ:
  Phiên 1: [laptop, mouse] → gợi ý [A, B, C] (có thể là "đầu chuột" hay gì đó lạ)
  Phiên 2: [dog_food, cat_toy] → gợi ý [A, B, C] (cũng không liên quan)
  
→ Recall = 0.0056 (chỉ ~0.56% trường hợp target trùng item phổ biến)
```

---

## 5. Thiết Kế Benchmark

### Script Chính: benchmark.py

**Pipeline:**
```python
[1/4] Load Data (train + test)
↓
[2/4] Load GRU4Rec (pretrained checkpoint)
      Extract embeddings for LastItem
↓
[3/4] Train Baselines
      - MostPopular.fit(train_data) → đếm tần suất
      - LastItem.fit(train_data, embeddings=extracted_embs)
↓
[4/4] Evaluate
      - GRU4Rec: batch_eval() → recall, MRR, coverage, ILD, diversity
      - MostPopular: run_baseline_eval() → recall, MRR
      - LastItem: run_baseline_eval() → recall, MRR
↓
Format Report (bảng so sánh)
```

### Công Thức Đánh Giá

**Recall@K:**
$$\text{Recall@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\text{target}_i \in \text{rec}_i^K)$$

**MRR@K (Mean Reciprocal Rank):**
$$\text{MRR@K} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}(\text{target}_i)} \quad \text{(nếu rank} \leq K\text{)}$$

**Ý tưởng:**
- Recall: Tỷ lệ đúng (target có trong top-K)
- MRR: Vị trí target trong danh sách (rank cao → MRR cao)

---

## 6. Kết Luận

### Những Điểm Quan Trọng

| Aspect | MostPopular | LastItem | GRU4Rec |
|--------|------------|----------|---------|
| **Nhân**: Lịch sử phiên | [-] | [-] (chỉ 1 item) | [+] (cả chuỗi) |
| **Học Temporal Pattern** | [-] | [-] | [+] |
| **Similarity** | Toàn cục | Pairwise | Deep (GRU state) |
| **Recall@20** (Yoochoose) | 0.0056 | 0.3090 | **0.6281** |
| **Độ Phức Tạp** | O(K) | O(N·d) | O(N·h) batch |
| **Cá Nhân Hóa** | [-][-][-] | [+] (yếu) | [+][+][+] (mạnh) |

### Tại Sao Thiết Kế Baseline Quan Trọng?

1. **Chứng minh GRU4Rec học được gì:** Nếu GRU4Rec chỉ tốt hơn MostPopular 10%, có thể là do data bias, không phải mô hình tốt. Nhưng vượt 100×? → GRU4Rec thực sự học được pattern.

2. **Xác định improvements thực tế:** LastItem ≈ GRU4Rec → learning không cần GRU, đơn giản similarity là đủ. Nhưng LastItem ≈ nửa GRU → GRU thực sự có giá trị.

3. **Benchmark công bằng:** Dùng embedding từ GRU cho LastItem (không phải random) → so sánh "apple-to-apple", chứ không so sánh GRU hiện đại vs LastItem lỗi thời.

---

## 7. Takeaway

> **GRU4Rec thắng baseline vì:**
> 1. Xử lý **cả chuỗi** item (không chỉ một item)
> 2. Học **thứ tự temporal** qua recurrent state
> 3. Phát hiện **long-range dependency** (laptop → cần phần mềm sau vài click)
> 4. Tạo **deep representation** qua hidden state (k đơn thuần similarity)
>
> **Baseline quan trọng vì:**
> - Là mốc so sánh công bằng, đơn giản
> - Xác nhận giả thuyết "cá nhân hóa theo phiên là cần thiết"
> - Giúp đánh giá giá trị thực sự của độ phức tạp mô hình
