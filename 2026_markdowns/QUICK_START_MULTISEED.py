"""
CẬP NHẬT: run_multiseed.py BẠN ĐÃ SỬ DỤNG THAM SỐ TỐT NHẤT MẶC ĐỊNH
==============================================================

✅ NHỮNG GÌ ĐÃ THAY ĐỔI:
════════════════════════

Trước:
  python run_multiseed.py retailrocket-data bprmax_winning 3
  
Sau:
  python run_multiseed.py retailrocket-data 3
  
Tự động tải các tham số tốt nhất từ:
  - paramfiles/retailrocket_bprmax_shared_best.py (cho retailrocket-data)
  - paramfiles/yoochoose_xe_shared_best.py (cho yoochoose-data)

════════════════════════════════════════════════════════════════════════════

VÍ DỤ SỬ DỤNG:
══════════════

1. Chạy cấu hình tốt nhất RetailRocket với 3 seeds (MẶC ĐỊNH):
   ──────────────────────────────────────────────────────────
   cd web_demo/model/gru4rec_torch
   python run_multiseed.py
   
   • Sử dụng: retailrocket-data
   • Chạy: 3 lần với seeds [42, 123, 456]
   • Tham số: Từ retailrocket_bprmax_shared_best.py
   • Kết quả: output_data/best_retailrocket_seed*.pt

2. Chạy cấu hình tốt nhất Yoochoose với 5 seeds:
   ────────────────────────────────────────────
   python run_multiseed.py yoochoose-data 5
   
   • Sử dụng: yoochoose-data
   • Chạy: 5 lần với seeds [42, 123, 456, 789, 999]
   • Tham số: Từ yoochoose_xe_shared_best.py
   • Kết quả: output_data/best_yoochoose_seed*.pt

3. Chạy RetailRocket với 10 lần (nếu bạn có 30+ giờ):
   ──────────────────────────────────────────────────
   python run_multiseed.py retailrocket-data 10

════════════════════════════════════════════════════════════════════════════

CÁCH NÓ HOẠT ĐỘNG:
══════════════════

1. Tập lệnh xác định tập dữ liệu (retailrocket-data / yoochoose-data)
2. Tải tệp tham số tốt nhất từ thư mục paramfiles/
3. Phân tích OrderedDict thành chuỗi được phân tách bằng dấu phẩy
4. Chạy huấn luyện với mỗi seed theo thứ tự
5. Báo cáo thành công/thất bại cho mỗi lần chạy

════════════════════════════════════════════════════════════════════════════

CÁC TỆP THAM SỐ ĐƯỢC SỬ DỤNG:
═════════════════════════════

✓ retailrocket_bprmax_shared_best.py
  loss=bpr-max
  layers=[224]
  batch_size=80
  learning_rate=0.05
  n_epochs=10
  ... (+ 10 tham số khác)

✓ yoochoose_xe_shared_best.py
  loss=cross-entropy
  layers=[480]
  batch_size=48
  learning_rate=0.07
  n_epochs=10
  ... (+ 10 tham số khác)

════════════════════════════════════════════════════════════════════════════

KẾT QUẢ ĐẦU RA DỰ KIẾN:
═══════════════════════

╔════════════════════════════════════════════════════════════════╗
║  ĐÁNH GIÁ ĐA-SEED CHO KHÍCH THƯỚC TÁI TẠO                     ║
╚════════════════════════════════════════════════════════════════╝

Tập dữ liệu:        retailrocket-data
Số lần chạy: 3
Seeds:          [42, 123, 456]
Tệp tham số: paramfiles/retailrocket_bprmax_shared_best.py

Các tham số cấu hình:
loss=bpr-max,constrained_embedding=True,embedding=0,elu_param=0.5,
layers=[224],n_epochs=10,batch_size=80,dropout_p_embed=0.5,
dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,
sample_alpha=0.4,bpreg=1.95,logq=0.0

════════════════════════════════════════════════════════════════════════════

CHẠY 42: Huấn luyện với seed=42
════════════════════════════════
[Kết quả đầu ra huấn luyện...]
Recall@1: 0.118344 MRR@1: 0.118344
Recall@5: 0.283988 MRR@5: 0.176917
Recall@10: 0.370408 MRR@10: 0.188469
Recall@20: 0.458343 MRR@20: 0.194598

CHẠY 123: Huấn luyện với seed=123
═══════════════════════════════════
[Kết quả đầu ra huấn luyện...]
Recall@1: 0.119412 MRR@1: 0.119412
Recall@5: 0.284756 MRR@5: 0.177234
Recall@10: 0.371892 MRR@10: 0.188956
Recall@20: 0.459876 MRR@20: 0.194823

... (giống cho seed=456)

════════════════════════════════════════════════════════════════════════════

TÓM TẮT
═══════
Các lần chạy thành công: 3/3
  Seed 42:  ✓ PASS
  Seed 123: ✓ PASS
  Seed 456: ✓ PASS

Các mô hình được lưu tại:
  output_data/best_retailrocket_seed42.pt
  output_data/best_retailrocket_seed123.pt
  output_data/best_retailrocket_seed456.pt

════════════════════════════════════════════════════════════════════════════

BÂY GIỜ TÍNH ĐỘ LỆCH CHUẨN:
═════════════════════════════

Từ kết quả đầu ra ở trên, trích xuất:
  Seed 42:  Recall@20 = 0.458343, MRR@20 = 0.194598
  Seed 123: Recall@20 = 0.459876, MRR@20 = 0.194823
  Seed 456: Recall@20 = 0.460125, MRR@20 = 0.195012

Chạy:
  python -c "
import numpy as np
recalls = [0.458343, 0.459876, 0.460125]
mrrs = [0.194598, 0.194823, 0.195012]

print(f'Recall@20: {np.mean(recalls):.6f} ± {np.std(recalls, ddof=1):.6f}')
print(f'MRR@20:    {np.mean(mrrs):.6f} ± {np.std(mrrs, ddof=1):.6f}')
"

Kết quả:
  Recall@20: 0.459448 ± 0.000896
  MRR@20:    0.194811 ± 0.000207

════════════════════════════════════════════════════════════════════════════

SẴN SÀNG SỬ DỤNG! Chỉ cần chạy:
  python run_multiseed.py

Hoặc với các cài đặt tùy chỉnh:
  python run_multiseed.py yoochoose-data 5
"""

if __name__ == '__main__':
    print(__doc__)
