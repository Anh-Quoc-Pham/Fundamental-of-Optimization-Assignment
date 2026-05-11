import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Premium styling for consistency
def set_premium_style():
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.grid.axis'] = 'y'
    plt.rcParams['grid.color'] = '#E0E0E0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['text.color'] = '#333'
    plt.rcParams['axes.labelcolor'] = '#333'

set_premium_style()

out_dir = "report_assets/section3"
os.makedirs(out_dir, exist_ok=True)

# 1. Inventory Status Distribution
inv_df = pd.read_csv("results/inventory_analysis.csv")

if 'inventory_status' in inv_df.columns:
    status_counts = inv_df['inventory_status'].value_counts()
    
    excess_count = status_counts.get('Excess', 0)
    needed_count = status_counts.get('Needed', 0)
    balanced_count = status_counts.get('Balanced', 0)
else:
    # Calculate from other files if inventory_status column is missing
    excess_df = pd.read_csv("results/excess_inventory.csv")
    needed_df = pd.read_csv("results/needed_inventory.csv")
    excess_count = len(excess_df)
    needed_count = len(needed_df)
    balanced_count = len(inv_df) - excess_count - needed_count

labels = ['Excess\n(Thừa)', 'Balanced\n(Cân bằng)', 'Needed\n(Thiếu)']
counts = [excess_count, balanced_count, needed_count]
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, counts, color=colors, width=0.5, zorder=3)

ax.set_ylabel('Số tổ hợp cửa hàng-sản phẩm')
ax.set_title('PHÂN BỐ TRẠNG THÁI TỒN KHO')
ax.set_ylim(0, max(counts) * 1.25)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{int(height)}',
            ha='center', va='bottom', fontweight='bold', fontsize=12, color='white',
            bbox=dict(facecolor='#2C3E50', edgecolor='none', boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.savefig(f"{out_dir}/fig_3_inventory_status_distribution.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{out_dir}/fig_3_inventory_status_distribution.pdf", bbox_inches='tight')
plt.close()

# 2. Store Locations
stores_df = pd.read_csv("data/stores.csv")

has_locations = 'latitude' in stores_df.columns and 'longitude' in stores_df.columns
if has_locations:
    # Use standard seaborn-like style but with matplotlib for scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Hide grid for map
    ax.grid(False)
    # Re-enable spines to form a bounding box for the "map"
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    for spine in ax.spines.values():
        spine.set_color('#E0E0E0')
        spine.set_linewidth(1.5)
        
    cities = stores_df['city'].unique() if 'city' in stores_df.columns else ['All']
    cmap = plt.get_cmap('Set2')
    city_colors = {city: cmap(i % 8) for i, city in enumerate(cities)}
    
    for city in cities:
        subset = stores_df[stores_df['city'] == city] if 'city' in stores_df.columns else stores_df
        ax.scatter(subset['longitude'], subset['latitude'], 
                   c=[city_colors[city]], label=city, 
                   s=150, alpha=0.8, edgecolors='white', linewidth=1.5, zorder=3)
                   
    for _, row in stores_df.iterrows():
        ax.annotate(str(row['store_id']), (row['longitude'], row['latitude']), 
                    xytext=(4, 4), textcoords='offset points', 
                    fontsize=9, fontweight='bold', color='#2C3E50', zorder=4)
                    
    ax.set_xlabel('Longitude (Kinh độ)')
    ax.set_ylabel('Latitude (Vĩ độ)')
    ax.set_title('PHÂN BỐ ĐỊA LÝ CỬA HÀNG', pad=15)
    
    if len(cities) > 1:
        ax.legend(title="Thành phố", loc='best', frameon=True, facecolor='white')
        
    ax.text(0.5, -0.15, "Ghi chú: Tọa độ cửa hàng được dùng để xây dựng ma trận khoảng cách và chi phí vận chuyển.", 
            ha='center', va='center', transform=ax.transAxes, fontsize=10, style='italic', color='#7F8C8D')
            
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig_3_store_locations.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/fig_3_store_locations.pdf", bbox_inches='tight')
    plt.close()

# 3. Summary File
summary_content = f"""# Data Visuals Summary cho Section 3.4

## A. Files đã tạo
* `fig_3_inventory_status_distribution.png` & `.pdf`
* {'`fig_3_store_locations.png` & `.pdf`' if has_locations else '(Không tạo được Store Locations do thiếu cột lat/lon)'}
* `section3_4_data_visuals_summary.md`

## B. Số liệu thực tế (Phân bố trạng thái tồn kho)
* **Excess (Thừa):** {excess_count} (Kỳ vọng: 353)
* **Needed (Thiếu):** {needed_count} (Kỳ vọng: 87)
* **Balanced (Cân bằng):** {balanced_count} (Kỳ vọng: 160)

*Trạng thái số liệu:* {'Khớp hoàn toàn với kỳ vọng!' if excess_count == 353 and needed_count == 87 and balanced_count == 160 else 'Có sai lệch so với kỳ vọng ban đầu.'}

## C. Thông tin Store Locations
* {'Đã tạo thành công bản đồ phân bố địa lý các cửa hàng dựa trên longitude/latitude.' if has_locations else 'Không tìm thấy tọa độ địa lý trong file stores.csv.'}
* Tọa độ cửa hàng được dùng để xác định khoảng cách và tính chi phí vận chuyển trong Transportation Simplex.

## D. Gợi ý Caption cho LaTeX (Tiếng Việt)
1. `fig_3_inventory_status_distribution`: *Hình 3.x: Phân bố trạng thái tồn kho của các tổ hợp cửa hàng-sản phẩm trước khi điều chuyển.*
2. {'`fig_3_store_locations`: *Hình 3.y: Phân bố địa lý các cửa hàng trên hệ thống mạng lưới bán lẻ.*' if has_locations else ''}
"""

with open(f"{out_dir}/section3_4_data_visuals_summary.md", "w", encoding="utf-8") as f:
    f.write(summary_content)

print("Section 3.4 visualization script executed successfully.")
