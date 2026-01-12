"""
LÀM RÕ PROTOCOL ĐÁNH GIÁ
==================================

Câu hỏi: Mỗi phiên kiểm thử có bao nhiêu bước dự đoán?
         Sử dụng toàn bộ phiên hay chỉ mục cuối cùng?

ĐÁP ÁN: NHIỀU DỰ ĐOÁN MỖI PHIÊN (Một dự đoán cho mỗi mục trong phiên)
=========================================================================

GIẢI THÍCH CHI TIẾT
--------------------

1. CẤU THÀNH CỦA PHIÊN:
   - Mỗi phiên chứa một chuỗi các mục: [item_1, item_2, ..., item_n]
   - Ví dụ: phiên có thể có 5 mục [a, b, c, d, e]

2. CÁC BƯỚC DỰ ĐOÁN MỖI PHIÊN:
   - Số lượng dự đoán = (độ dài phiên - 1)
   - Đối với phiên có 5 mục: 4 bước dự đoán
   
3. CÁCH THỰC HIỆN DỰ ĐOÁN:
   
   Bước 1: Đầu vào=[item_1],              Mục tiêu=item_2     (Dự đoán mục thứ 2)
   Bước 2: Đầu vào=[item_1, item_2],      Mục tiêu=item_3     (Dự đoán mục thứ 3)
   Bước 3: Đầu vào=[item_1, item_2, item_3], Mục tiêu=item_4  (Dự đoán mục thứ 4)
   Bước 4: Đầu vào=[item_1, item_2, item_3, item_4], Mục tiêu=item_5 (Dự đoán mục thứ 5)

4. BẰNG CHỨNG TRONG CODE:
   
   Từ SessionDataIterator.__call__() trong gru4rec_pytorch.py (dòng 400-445):
   
   - minlen = (end - start).min()  # Tính độ dài phiên tối thiểu
   
   - for i in range(minlen - 1):   # Lặp từ 0 đến (độ dài - 2)
         in_idx = out_idx  # Đầu vào: các mục hiện tại cho đến điểm này
         out_idx = torch.tensor(self.data_items[start + i + 1])  # Mục tiêu: mục TIẾP THEO
         yield in_idx, out_idx
   
   Điều này tạo ra (độ dài phiên - 1) cặp huấn luyện cho mỗi phiên.

5. CÁC CHỈ SỐ ĐÁNH GIÁ:
   
   Từ batch_eval() trong evaluation.py (dòng 202-227):
   
   - Đối với mỗi bước dự đoán:
     * Lấy điểm mô hình cho tất cả các mục: oscores
     * Lấy điểm mục thực (sự thật gốc): tscores = torch.diag(oscores[out_idxs])
     * Tính thứ hạng của mục thực so với tất cả các mục khác
     * Đếm nếu thứ hạng <= K (K=1, 5, 10, 20)
     * Tính Recall@K = (đếm / tổng_dự_đoán) * 100
     * Tính MRR@K = trung bình(1/thứ_hạng cho tất cả các dự đoán có thứ_hạng <= K)
   
   - TỔNG DỰ ĐOÁN = tổng (độ dài phiên - 1) trên tất cả các phiên kiểm thử
   
   Ví dụ từ diary.md (kết quả ngày 28 tháng 12):
   - Dữ liệu kiểm thử gốc: 44,910 sự kiện
   - Đây KHÔNG phải 44,910 bước dự đoán!
   - Mỗi độ dài phiên >= 2 tạo ra (độ dài phiên - 1) dự đoán
   - Vì vậy tổng dự đoán ≈ 44,129 (các sự kiện kiểm thử được lọc - num_sessions)

6. TÓM TẮT:
   
   ✅ Sử dụng TOÀN BỘ phiên (tất cả các mục ngoại trừ mục cuối cùng)
   ✅ Dự đoán MỤC TIẾP THEO ở mỗi bước
   ✅ Nhiều dự đoán mỗi phiên (một dự đoán cho mỗi mục, ngoại trừ mục đầu tiên và cuối cùng)
   ✅ KHÔNG chỉ dự đoán mục cuối cùng một lần cho mỗi phiên
   
   Đây là PROTOCOL ĐÁNH GIÁ TIÊU CHUẨN cho các hệ thống đề xuất dựa trên phiên
   (các mô hình dựa trên RNN như GRU4Rec sử dụng toàn bộ lịch sử cho đến điểm hiện tại).

7. CÁC HẠN CHẾ QUAN TRỌNG CHO RQ:
   
   RQ1-3 nên làm rõ:
   - Đánh giá được thực hiện trên cơ sở mỗi dự đoán, không phải trên cơ sở mỗi phiên
   - Các chỉ số tập hợp trên tất cả các dự đoán (độ dài phiên - 1)
   - Nếu một phiên có 10 mục: 9 bước dự đoán đóng góp vào các chỉ số cuối cùng
"""

if __name__ == "__main__":
    print(__doc__)
