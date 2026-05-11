# Thực nghiệm Optimal Transport trong Domain Adaptation (Mini Experiment)

## Cấu hình Dataset
- **Loại**: 2D Gaussian Blobs (scikit-learn `make_blobs`).
- **Số lượng**: 600 samples cho Source, 600 samples cho Target (300 samples/class).
- **Domain Shift**: Quay 20 độ, dịch chuyển rất mạnh để Source Only baseline thất bại hoàn toàn.
- **Random Seed**: Fix ở 42 (Source) và 4242 (Target).

## Phương pháp
- **Classifier**: SVC RBF.
- Thư viện POT không khả dụng. Thuật toán OT được tự implement hoàn toàn.
- **Exact OTDA**: Mô phỏng bằng Linear Sum Assignment. Lưu ý: Đây là trường hợp đặc biệt giải chính xác bài toán OT khi số lượng mẫu của source và target bằng nhau và trọng số empirical phân bố đều (1/n). Phương pháp này không áp dụng tổng quát cho mọi bài toán OT.
- **Sinkhorn OTDA**: Entropy regularized (epsilon=0.05). Tự implement ổn định hóa bằng log-domain (chống tràn số).

## Kết quả
- **Source-only Accuracy**: 50.0%
- **Exact OTDA Accuracy**: 100.0%
- **Sinkhorn OTDA Accuracy**: 100.0%

## Quan sát & Caveat
- Accuracy bão hòa (100%) trên toy dataset này đối với các cấu hình OTDA. Việc điều chỉnh epsilon chủ yếu ảnh hưởng đến độ mượt và cost của coupling (alignment cost) hơn là cải thiện thêm về accuracy.
- Trong thí nghiệm minh họa này, OTDA cải thiện accuracy từ 50.0% lên 100.0% do source và target có cấu trúc lớp tương thích và domain shift được thiết kế rõ ràng. Đây là toy 2D experiment dùng để minh họa cơ chế OTDA, không phải benchmark tổng quát.
- Alignment Cost không hoàn toàn so sánh trực tiếp được giữa exact và Sinkhorn do ảnh hưởng của hàm mục tiêu có chứa thành phần regularization trong Sinkhorn.

## Các File Xuất
- `fig_4_otda_pipeline.[png/pdf]`
- `fig_4_domain_shift_before.[png/pdf]`
- `fig_4_transport_plan_sample.[png/pdf]`
- `fig_4_domain_shift_after.[png/pdf]`
- `fig_4_accuracy_comparison.[png/pdf]`
- `fig_4_sinkhorn_sensitivity.[png/pdf]`
- `table_4_otda_results.tex`
