# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Phan Văn Nghị |
| MSSV | 2A202602632 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/vannghj/K4-L3-DAY21-PhanVanNghi-2A202602632-CI-CD-for-AI-Systems |
| Ngày nộp | 08/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.874 |
| 4 | 200 | 0.05 | 3 | 0.7014 | 0.874 |
| 5 | 100 | 0.2 | 5 | 0.7207 | 0.876 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.2`, `max_depth=5`.

**Lý do:** Bộ này có F1 cao nhất (0.7207). Lần có accuracy cao nhất (lần 1, 0.878) lại chỉ đạt F1 0.7109, tức chọn theo accuracy sẽ ra mô hình bắt lớp thu nhập cao kém hơn. Accuracy chỉ dao động 0.846 - 0.878, còn F1 dao động 0.605 - 0.721, nên F1 phân biệt cấu hình rõ hơn. Về đánh đổi, giảm `learning_rate` xuống 0.05 làm F1 giảm, kể cả khi tăng `n_estimators` lên 200 để bù (lần 4).

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu thuộc lớp thu nhập > 50K, nên mô hình luôn đoán "thu nhập thấp" vẫn đạt accuracy 0.752 dù không tìm ra được ai thu nhập cao; quality gate theo accuracy có thể cho mô hình đó đi qua. F1 lớp dương kết hợp precision và recall của chính lớp thu nhập cao, nên mô hình kia có F1 = 0. Không dùng `average="weighted"` hay `"macro"` vì chúng trộn F1 của lớp đa số (0.920) vào: với mô hình đã chọn, F1 lớp dương là 0.721 nhưng `macro` cho 0.821 và `weighted` cho 0.871, che mất việc bỏ sót lớp dương. Ngưỡng này đã được kiểm chứng: khi push bộ params yếu (`n=50, lr=0.05, depth=2`), mô hình vẫn đạt accuracy 0.842 nhưng F1 chỉ 0.5907, nên Quality Gate thất bại và Release bị bỏ qua (ảnh `07-quality-gate-chan.png`).

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `import mlflow` lỗi `pkg_resources`, rồi lỗi SQLAlchemy khi dùng SQLite. | mlflow 2.13 không tương thích Python 3.14, setuptools và SQLAlchemy 2.1 mới. | Dùng venv Python 3.12, ghim `setuptools<70` và `sqlalchemy<2.1`. |
| Không tạo được service account key. | Organization GCP bật mặc định `iam.disableServiceAccountKeyCreation`. | Override chính sách chỉ ở project của lab; service account vẫn chỉ có `storage.objectAdmin` trên bucket. |
| `git push` không kích hoạt GitHub Actions. | Repo là fork, GitHub không chạy workflow theo push cho đến khi bật Actions. | Chạy Bước 2 bằng `workflow_dispatch`, bật Actions, revert rồi tạo lại commit dữ liệu Bước 3. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7207 | 0.876 |
| Bước 3 (thêm `train_batch2`) | 0.7297 | 0.880 |

**Nhận xét:** Gấp đôi dữ liệu chỉ làm F1 tăng 0.009 và accuracy tăng 0.004, tức 2 mẫu đúng thêm trên 500 mẫu holdout, nằm trong mức dao động. Lý do là `train_batch2` có cùng phân phối với `train_batch1` nên mang ít thông tin mới. Điều Bước 3 chứng minh là quy trình: commit dữ liệu tự kích hoạt cả bốn job và model mới được đưa lên VM mà không cần thao tác thủ công.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 1 - Job Train ghi MLflow lên DagsHub qua secrets `MLFLOW_TRACKING_*`; run `righteous-bee-621` có đủ f1_score, accuracy, best_threshold (ảnh 06).
- [x] Bonus 2 - Ngưỡng 0.30 cho F1 0.7452 so với 0.7297 ở ngưỡng 0.5 (chọn trên chính holdout nên hơi lạc quan; API vẫn dùng 0.5).
- [x] Bonus 3 - `detail.txt`: lớp thu nhập cao có precision 0.827, recall 0.653 (bỏ sót 43, gán nhầm 17). Nếu dùng để tìm khách cho sản phẩm tài chính cao cấp thì bỏ sót tốn kém hơn, nên ưu tiên recall, ví dụ hạ ngưỡng như Bonus 2.
- [x] Bonus 4 - Model mới lên `candidate/`, chỉ promote sang `current/` khi F1 mới ≥ F1 cũ; params `n=200, lr=0.05, depth=3` (F1 0.6957 < 0.7297) bị hủy triển khai (ảnh 08).
- [x] Bonus 5 - Tỷ lệ lớp dương 24.78%, lệch không quá 5 điểm nên không cảnh báo; `positive_rate` có trong `report.json`.
