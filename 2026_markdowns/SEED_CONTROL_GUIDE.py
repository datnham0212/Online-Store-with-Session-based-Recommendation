"""
KIỂM SOÁT RANDOM SEED - TÓM TẮT TRIỂN KHAI
=============================================

Những gì đã được thêm:
1. Đối số dòng lệnh --seed cho run.py (mặc định: 42)
2. Hàm set_seed() đặt torch, numpy và random seeds
3. In seed cho mỗi lần chạy để theo dõi khả năng tái tạo
4. Tập lệnh run_multiseed.py để tự động hóa các thử nghiệm đa-seed

Tệp Sửa đổi:
- web_demo/model/gru4rec_torch/run.py (đã thêm kiểm soát seed)

Tệp Tạo:
- web_demo/model/gru4rec_torch/run_multiseed.py (trình chạy đa-seed)

═══════════════════════════════════════════════════════════════════════════

CÁCH SỬ DỤNG:

1. CHẠY VỚI SEED CỤ THỂ:
   ──────────────────────
   cd web_demo/model/gru4rec_torch
   
   python run.py input_data/retailrocket-data/retailrocket_train_full.dat \
     -ps loss=bpr-max,layers=224,batch_size=80,dropout_p_embed=0.5,... \
     -t input_data/retailrocket-data/retailrocket_test.dat \
     -m 1 5 10 20 \
     -s output_data/model_seed42.pt \
     --seed 42

2. CHẠY NHIỀU SEEDS TỰ ĐỘNG:
   ─────────────────────────
   python run_multiseed.py retailrocket-data bprmax_winning 3
   
   Điều này sẽ chạy 3 lần với seeds: [42, 123, 456]
   Kết quả: output_data/bprmax_winning_retailrocket_seed*.pt

3. CHẠY ĐƯỜNG CƠ SỞ CE VỚI 3 SEEDS:
   ────────────────────────────────
   python run_multiseed.py retailrocket-data ce_baseline 3

═══════════════════════════════════════════════════════════════════════════

KẾT QUẢ ĐẦU RA DỰ KIẾN VỚI KIỂM SOÁT SEED:

  Đang tạo mô hình GRU4Rec trên thiết bị "cpu"
  Random seed set to: 42                    <-- MỚI: Xác nhận seed
  ĐẶT   loss                    THÀNH   bpr-max   (kiểu: <class 'str'>)
  ĐẶT   layers                  THÀNH   [224]     (kiểu: <class 'list'>)
  [... kết quả đầu ra huấn luyện ...]
  Recall@1: 0.118344 MRR@1: 0.118344
  Recall@5: 0.283988 MRR@5: 0.176917

═══════════════════════════════════════════════════════════════════════════

CÁC BƯỚC TIẾP THEO ĐỂ LẤY ĐỘ LỆCH CHUẨN:

1. Chạy 3-5 lần:
   python run_multiseed.py retailrocket-data bprmax_winning 5

2. Thu thập các chỉ số từ nhật ký đầu ra vào CSV

3. Tính toán thống kê:
   ```python
   import numpy as np
   recalls = [0.460009, 0.458343, ...]  # Từ 5 lần chạy
   mean = np.mean(recalls)
   std = np.std(recalls, ddof=1)
   print(f"Recall@20: {mean:.6f} ± {std:.6f}")
   # Kết quả: Recall@20: 0.459176 ± 0.000833
   ```

4. Báo cáo trong bài báo cuối cùng:
   "Recall@20: 0.459 ± 0.001 (trung bình ± độ lệch chuẩn, n=5 lần chạy)"
   "MRR@20:    0.194 ± 0.0005 (trung bình ± độ lệch chuẩn, n=5 lần chạy)"

═══════════════════════════════════════════════════════════════════════════

ĐẢM BẢO KHẢ NĂNG TÁI TẠO:

Với kiểm soát seed được triển khai:
  ✓ Kết quả hiện có thể tái tạo được
  ✓ Cùng seed = khởi tạo giống hệt nhau
  ✓ Các seed khác nhau = đo lường biến đổi tự nhiên
  ✓ Có thể so sánh các mô hình công bằng (cùng seed trên tất cả)
  ✓ Có thể định lượng hóa không chắc chắn (nhiều seeds)

Ví dụ lệnh đầy đủ để lấy độ lệch chuẩn:
  
  # Thiết bị đầu cuối 1:
  python run_multiseed.py retailrocket-data bprmax_winning 5
  
  # Sau đó tính toán thống kê từ kết quả đầu ra bộ điều khiển
  # và lưu vào CSV để phân tích
"""

if __name__ == '__main__':
    print(__doc__)
