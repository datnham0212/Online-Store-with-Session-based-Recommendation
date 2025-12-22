# Kế Hoạch Thực Hiện Phase 6 - Các Hành Động Tiếp Theo

## 5 Hành Động Chính

### 1. Điều Tra Khoảng Cách Recall (train_valid vs train_full)

**Vấn đề:** Tại sao train_full đạt 0.605 khi train_valid chỉ 0.588?

**Kết luận sơ bộ:** Không phải overfitting, mà do:
- train_full có nhiều dữ liệu hơn (82.6% lớn hơn)
- Mô hình học các mẫu item phong phú hơn
- Dữ liệu thực tế tốt hơn cho learning

**Kiểm tra cần chạy:**
- Phân tích kích thước dữ liệu ảnh hưởng (T6.2b): 50% train_full
- Xác nhận: Recall tăng theo kích thước dataset

---

### 2. Thử Nghiệm Thêm RetailRocket với BPR-max

**Hiện trạng:** BPR-max baseline (T5.3b) = Recall@20 0.408

**Tối ưu hóa:**
- Tăng epochs từ 3 → 5 (T6.3b)
- Tăng capacity: layers từ 224 → 256 (T6.3c)
- Dự kiến cải thiện 1-3%

**Script:** `run_tests_phase6.bat` - T6.3b & T6.3c

---

### 3. Giảm Thời Gian Huấn Luyện (Early Stopping)

**Hiện trạng:** 5 epochs = 249s, 6 epochs = 2753s (train_full)

**Chiến lược:**
- 3 epochs: Dự kiến 130s, Recall@20 ≈ 0.587 (tiết kiệm 48%)
- 4 epochs: Dự kiến 175s, Recall@20 ≈ 0.5875 (tiết kiệm 30%)

**Script:** `run_tests_phase6.bat` - T6.5a & T6.5b

**Kỳ vọng:** Early stopping tại epoch 3-4 đủ, tiết kiệm thời gian rất lớn

---

### 4. Benchmark Độ Trễ Inference (p99 < 100ms)

**Mục đích:** Xác thực production-ready

**Công cụ:** `benchmark_latency.py` - Đo latency per-session

**Chỉ số:**
- p50: Median latency
- p95: 95th percentile
- **p99: 99th percentile (mục tiêu < 100ms)**
- p99.9: Extreme outliers

**Script:** `run_latency_benchmark.bat`

**Chiến lược:**
- Nếu layers=112 p99 < 100ms: Dùng layers=112
- Nếu layers=112 p99 > 100ms: Dùng layers=96 (nhanh hơn 6%)

---

### 5. So Sánh TOP-1 Loss (Đầy Đủ)

**Hiện trạng:** T2.1c & T2.1d (TOP-1 gốc) hoàn toàn thất bại trên Yoochoose

**Giả thuyết:** Cấu hình không tối ưu, không phải mất mát TOP-1

**Kiểm tra:**

#### 5a. TOP-1 Tuned trên Yoochoose
```
Cấu hình:
  - Loss: top1
  - Layers: 128
  - Batch: 32
  - LR: 0.05
  - Momentum: 0.3
  - Epochs: 5
```
**Script:** `run_tests_phase6.bat` - T6.7a

#### 5b. TOP-1 Max Tuned trên Yoochoose
```
Cấu hình:
  - Loss: top1-max
  - Layers: 128
  - Batch: 64
  - LR: 0.1
  - Momentum: 0.0
  - Epochs: 5
```
**Script:** `run_tests_phase6.bat` - T6.7b

#### 5c. TOP-1 trên RetailRocket
```
Cấu hình:
  - Loss: top1
  - Layers: 100
  - Batch: 32
  - LR: 0.05
  - Epochs: 3
```
**Script:** `run_tests_phase6.bat` - T6.8a

#### 5d. TOP-1 Max trên RetailRocket
```
Cấu hình:
  - Loss: top1-max
  - Layers: 100
  - Batch: 64
  - LR: 0.1
  - Epochs: 3
```
**Script:** `run_tests_phase6.bat` - T6.8b

**Kỳ vọng:**
```
Yoochoose TOP-1 tuned:       Recall@20 > 0.4 (hoặc vẫn 0?)
RetailRocket TOP-1:           Recall@20 > 0.2 (hoặc vẫn 0?)
Kết luận: TOP-1 khả thi hay không, so sánh vs CE/BPR-max
```

---

## Danh Sách Chạy Test

### Thực Hiện Ngay (Ưu Tiên Cao)

```bash
# 1. Early stopping (tiết kiệm 48-30% thời gian)
.\run_tests_phase6.bat

# 2. Latency benchmark (xác thực p99 < 100ms)
.\run_latency_benchmark.bat

# 3. TOP-1 comparison (so sánh 4 loss functions)
# Đã bao gồm trong run_tests_phase6.bat
```

### Ước Tính Thời Gian

```
T6.5a (3 epochs):          ~130s
T6.5b (4 epochs):          ~175s
T6.7a (TOP-1 Yoochoose):   ~300s
T6.7b (TOP1-max Yoochoose):~350s
T6.8a (TOP-1 RR):          ~200s
T6.8b (TOP1-max RR):       ~250s
T6.3b (BPR-max RR, 5ep):   ~400s
T6.3c (BPR-max RR, 256):   ~500s

Latency benchmark:         ~300s (1000 samples per model)

Tổng cộng: ~2.5-3 giờ
```

---

## Kết Quả Mong Đợi

### Sau Phase 6

| Hành động | Kết quả dự kiến |
|-----------|-----------------|
| **Recall gap investigation** | Không phải overfitting, data size effect |
| **Early stopping** | Đủ 3-4 epochs, tiết kiệm 30-50% thời gian |
| **Latency p99** | < 100ms hoặc cần layers=96 |
| **TOP-1 tuned** | Viable (hoặc vẫn không hoạt động) |
| **RetailRocket tuning** | BPR-max tốt hơn 1-3% với epochs/capacity tăng |

### Production Config Cuối Cùng

```
Yoochoose (Recommended):
  - Loss: cross-entropy
  - Layers: 112
  - Batch: 128
  - Epochs: 3-4 (early stopped)
  - Dataset: train_full
  - Recall@20: 0.600+
  - Latency p99: < 100ms

RetailRocket (Alternative):
  - Loss: bpr-max (tuned)
  - Layers: 224+ (tùy chỉnh)
  - Batch: 64-80
  - Epochs: 3-5
  - Recall@20: 0.40-0.42
```

---

## Các File Tạo Ra

1. **PHASE6_NEXT_ACTIONS.md** - Kế hoạch chi tiết (đây là file này)
2. **run_tests_phase6.bat** - Script chạy tất cả tests (T6.3-T6.8)
3. **run_latency_benchmark.bat** - Benchmark latency per-session
4. **benchmark_latency.py** - Python script latency profiling

---

## Cách Chạy

### Chạy Phase 6 Tests

```bash
cd c:\Users\Admin\Documents\Research\Online Store with Session-based Recommendation\web_demo\model\gru4rec_torch

# Chạy tất cả: Early stopping, TOP-1, BPR-max tuning
.\run_tests_phase6.bat

# Chạy latency benchmark
.\run_latency_benchmark.bat
```

### Kiểm Tra Kết Quả

```bash
# Kết quả Phase 6
type output_data\phase6_results.log

# Latency benchmark
type output_data\latency_benchmark.txt
```

---

## Lưu Ý

- Tất cả scripts sử dụng **UTF-8 encoding** (PYTHONIOENCODING=utf-8)
- Tất cả tests sử dụng **CPU** (`-d cpu`)
- Kết quả tự động log vào `output_data/` folder
- Không có tạo markdown mới - chỉ update `README.md` sau hoàn thành

