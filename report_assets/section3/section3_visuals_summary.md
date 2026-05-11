# Visual Assets Summary cho Section 3

## A. Files generated
Tất cả các files đã được tạo thành công trong thư mục `report_assets/section3/`:
- `fig_3_pipeline.png` & `.pdf`
- `fig_3_pivot_loop.png` & `.pdf`
- `fig_3_cost_comparison.png` & `.pdf`
- `fig_3_coverage_unmet.png` & `.pdf`
- `fig_3_transfer_network.png` & `.pdf`
- `fig_3_transfer_network_top30.png` & `.pdf`
- `table_3_top_transfers.tex` & `.csv`
- `table_3_algorithm_comparison.tex`
- `table_3_validation.tex` & `validation_summary.csv`

## B. Commands run
Script sinh biểu đồ được chạy với Python sử dụng `matplotlib`, `networkx`, và `pandas`.
Dữ liệu đầu vào lấy trực tiếp từ thư mục `results/` và `data/` mà không cần chạy lại thuật toán.

## C. Main numbers
* **Transportation Simplex cost:** 29,101,871 VND
* **Rule-Based cost:** 29,406,546 VND
* **Cost Difference:** 304,674 VND
* **Cost reduction % (so với Rule-Based):** 1.04%
* **Coverage:** TS = 100% (891 units), Rule-Based = 100% (891 units)
* **Transfers:** TS = 93, Rule-Based = 92
* **Runtime:** TS = 0.04s, Rule-Based = 0.19s
* **Validation gap:** TS vs LP HiGHS = 0.00%

## D. Figures created and what each shows
* **fig_3_pipeline:** Flowchart mô tả luồng quy trình từ Data -> Inventory Need -> Transportation Model -> Result.
* **fig_3_pivot_loop:** Hình minh họa học thuật cho thao tác Pivot (entering cell `+`, subtract `-`, add `+`, leaving `-`). Không dùng số liệu thật, thuần túy dùng cho lý thuyết.
* **fig_3_cost_comparison:** Bar chart so sánh tổng chi phí vận chuyển. Thấy rõ TS tiết kiệm hơn Rule-Based một chút và match chính xác LP.
* **fig_3_coverage_unmet:** Stacked bar chart minh chứng rằng việc giảm chi phí của TS không đánh đổi bằng việc bỏ sót nhu cầu (Coverage = 100%).
* **fig_3_transfer_network (_top30):** Network graph hiển thị luồng hàng hóa giữa các cửa hàng. Node size tỉ lệ với tổng hàng luân chuyển qua cửa hàng, edge width tỉ lệ với số hàng luân chuyển trên tuyến đó.

## E. Tables created
* **table_3_top_transfers.tex:** LaTeX table format chuẩn `booktabs` hiển thị Top 10 chuyến hàng lớn nhất do thuật toán TS đề xuất, kèm chi phí và khoảng cách.
* **table_3_algorithm_comparison.tex:** So sánh tổng quan hiệu năng các thuật toán.
* **table_3_validation.tex:** LaTeX table báo cáo kết quả benchmark giữa TS tự code và Scipy LP HiGHS.

## F. Validation result
Trùng khớp 100%. `Transportation Simplex` đạt objective `29,101,871 VND`, tương đương chuẩn gốc `29,101,871 VND` với sai số `0.00%`.

## G. Cảnh báo / Caveats
- Dữ liệu hoàn toàn sạch, các biểu đồ đều tạo thành công. Không có issues nào đáng lưu ý. File `table_3_*.tex` chứa trực tiếp LaTeX source (`\toprule`, `\bottomrule` sẽ được render khi dùng package `booktabs` trong trình LaTeX của người dùng).
- Runtime được ghi chú lại từ kết quả chạy thực nghiệm trước đó, lưu ý người đọc đây chỉ là "trên bộ dữ liệu thử nghiệm này" và bị phụ thuộc cấu hình máy.

### Gợi ý Caption cho LaTeX (Tiếng Việt)
1. `fig_3_pipeline`: *Hình 3.1: Quy trình tổng thể từ phân tích dữ liệu bán hàng đến tối ưu hóa điều chuyển tồn kho.*
2. `fig_3_pivot_loop`: *Hình 3.2: Minh họa chu trình Pivot trong thuật toán đơn hình vận tải (Transportation Simplex).*
3. `fig_3_cost_comparison`: *Hình 3.3: So sánh tổng chi phí vận tải giữa phương pháp Rule-Based và Transportation Simplex.*
4. `fig_3_coverage_unmet`: *Hình 3.4: Tỷ lệ đáp ứng nhu cầu (Coverage) của các phương pháp điều chuyển.*
5. `fig_3_transfer_network`: *Hình 3.5: Mạng lưới điều chuyển hàng hóa tối ưu theo đề xuất của Transportation Simplex.*
