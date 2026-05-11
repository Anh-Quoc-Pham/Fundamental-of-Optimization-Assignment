import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.path import Path
import networkx as nx

# Ensure output directory exists
out_dir = "report_assets/section3"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Global Premium Styling Configuration
# -----------------------------------------------------------------------------
def set_premium_style():
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.titlesize'] = 16
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
    plt.rcParams['xtick.color'] = '#555555'
    plt.rcParams['ytick.color'] = '#555555'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'

set_premium_style()

# Custom Brand Colors
COLOR_RB = '#8E9EAB'       # Muted silver/blue for Rule-Based
COLOR_TS = '#FF6B6B'       # Vibrant coral for TS
COLOR_TS_ALT = '#4ECDC4'   # Vibrant teal
COLOR_BG = '#FFFFFF'       # White
COLOR_DARK_BG = '#1E1E2F'  # Sleek dark background for network
COLOR_NODE = '#00D2D3'     # Neon cyan for nodes
COLOR_EDGE = '#54A0FF'     # Soft blue for edges

# Load data
df_comp = pd.read_csv("results/algorithm_comparison.csv")
df_ts = pd.read_csv("results/transportation_simplex_transfers.csv")

# Extract metrics
df_comp_main = df_comp[df_comp['algorithm'].isin(['Transportation-Simplex', 'Rule-Based'])].copy()
df_comp_main.set_index('algorithm', inplace=True)
ts_cost = df_comp_main.loc['Transportation-Simplex', 'total_transport_cost']
rb_cost = df_comp_main.loc['Rule-Based', 'total_transport_cost']
cost_reduction_pct = (rb_cost - ts_cost) / rb_cost * 100
ts_units = df_comp_main.loc['Transportation-Simplex', 'covered_units']
rb_units = df_comp_main.loc['Rule-Based', 'covered_units']
ts_unmet = df_comp_main.loc['Transportation-Simplex', 'unmet_units']
rb_unmet = df_comp_main.loc['Rule-Based', 'unmet_units']
total_needed = df_comp_main.loc['Transportation-Simplex', 'total_needed_units']

# Helper to draw box with shadow
def draw_shadow_box(ax, x, y, width, height, text, facecolor, edgecolor, fontsize=12):
    # Shadow
    shadow = patches.FancyBboxPatch((x + 0.02, y - 0.02), width, height,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    edgecolor='none', facecolor='black', alpha=0.1)
    ax.add_patch(shadow)
    # Box
    rect = patches.FancyBboxPatch((x, y), width, height,
                                  boxstyle="round,pad=0.03,rounding_size=0.1",
                                  edgecolor=edgecolor, facecolor=facecolor, lw=2)
    ax.add_patch(rect)
    # Text
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', color='white' if mcolors.to_rgb(facecolor)[0] < 0.6 else '#333')

# -----------------------------------------------------------------------------
# 1. Pipeline Flowchart (Premium)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')

boxes = [
    "Dữ liệu Bán hàng & Tồn kho",
    "Tính toán Days of Supply",
    "Xác định Tồn kho Thừa / Thiếu",
    "Xây dựng Mô hình Vận tải",
    "Giải thuật Transportation Simplex",
    "Kế hoạch Điều chuyển & KPIs"
]

# Gradient colors for boxes
cmap = plt.get_cmap('ocean')
colors = [cmap(i) for i in np.linspace(0.8, 0.4, len(boxes))]

y_pos = np.linspace(0.85, 0.1, len(boxes))
box_width, box_height = 0.65, 0.08

for i, text in enumerate(boxes):
    draw_shadow_box(ax, 0.5 - box_width/2, y_pos[i] - box_height/2, box_width, box_height, 
                    text, facecolor=colors[i], edgecolor='white', fontsize=12)
    
    if i < len(boxes) - 1:
        # Elegant arrow
        arrow = patches.FancyArrowPatch((0.5, y_pos[i] - box_height/2 - 0.02),
                                        (0.5, y_pos[i+1] + box_height/2 + 0.02),
                                        arrowstyle='-|>', mutation_scale=20, 
                                        color='#A0A0A0', lw=3)
        ax.add_patch(arrow)

plt.title("QUY TRÌNH TỐI ƯU HÓA ĐIỀU CHUYỂN TỒN KHO\n", fontsize=18, fontweight='heavy', color='#2C3E50', pad=20)
ax.text(0.5, 0.96, "Từ dữ liệu bán lẻ đến mô hình vận tải tối ưu", ha='center', fontsize=13, color='#7F8C8D', style='italic')

plt.tight_layout()
plt.savefig(f"{out_dir}/fig_3_pipeline.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{out_dir}/fig_3_pipeline.pdf", bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 2. Pivot Loop Concept (Premium Cards)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

start_x, start_y = 0.2, 0.2
w, h = 0.25, 0.2
offset = 0.4

cells = [
    (start_x, start_y + offset, "Cộng θ (+)", '#4ECDC4', 'white'),           # Top left
    (start_x + offset, start_y + offset, "Trừ θ (-)", '#FF6B6B', 'white'),     # Top right
    (start_x + offset, start_y, "Ô vào cơ sở (+θ)", '#4ECDC4', 'white'),      # Bottom right (Entering)
    (start_x, start_y, "Ô rời cơ sở\n(- min θ)", '#FF6B6B', 'white')          # Bottom left (Leaving)
]

# Draw curved connections first so they go behind
for i in range(len(cells)):
    x1, y1 = cells[i][0] + w/2, cells[i][1] + h/2
    x2, y2 = cells[(i+1)%len(cells)][0] + w/2, cells[(i+1)%len(cells)][1] + h/2
    
    rad = 0.2 if i % 2 == 0 else -0.2
    arrow = patches.FancyArrowPatch((x2, y2), (x1, y1),
                                    connectionstyle=f"arc3,rad={rad}",
                                    arrowstyle='-|>', mutation_scale=25, 
                                    color='#2C3E50', lw=3, alpha=0.6)
    ax.add_patch(arrow)

# Draw cards
for x, y, text, color, textcolor in cells:
    draw_shadow_box(ax, x, y, w, h, text, facecolor=color, edgecolor='none', fontsize=12)

plt.title("CHU TRÌNH PIVOT TRONG TRANSPORTATION SIMPLEX", fontsize=16, fontweight='heavy', color='#2C3E50', pad=15)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.9)
plt.tight_layout()
plt.savefig(f"{out_dir}/fig_3_pivot_loop.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{out_dir}/fig_3_pivot_loop.pdf", bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 3. Cost Comparison Chart (Modern Bar)
# -----------------------------------------------------------------------------
set_premium_style()
fig, ax = plt.subplots(figsize=(7, 5.5))
labels = ['Rule-Based', 'Transportation Simplex']
costs_millions = [rb_cost / 1e6, ts_cost / 1e6]

bars = ax.bar(labels, costs_millions, color=[COLOR_RB, COLOR_TS], width=0.45, zorder=3)

ax.set_ylabel('Tổng chi phí (triệu VND)', fontsize=13, labelpad=10)
ax.set_title('SO SÁNH TỔNG CHI PHÍ VẬN CHUYỂN', fontsize=16, pad=20, color='#2C3E50')
ax.set_ylim(0, max(costs_millions) * 1.25)
ax.tick_params(axis='x', labelsize=13, pad=10)

# Add data labels with modern pill-boxes
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}M',
            ha='center', va='bottom', fontweight='bold', fontsize=12, color='white',
            bbox=dict(facecolor='#2C3E50', edgecolor='none', boxstyle='round,pad=0.4'))

# Impact annotation
ax.annotate(f"Giảm chi phí ≈ {cost_reduction_pct:.2f}%", 
            xy=(1, ts_cost / 1e6), xytext=(0.5, max(costs_millions)*1.15),
            arrowprops=dict(facecolor='#FF6B6B', shrink=0.05, width=2, headwidth=8, edgecolor='none'),
            ha='center', va='center', fontsize=13, fontweight='bold', color='white',
            bbox=dict(facecolor='#FF6B6B', edgecolor='none', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig(f"{out_dir}/fig_3_cost_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{out_dir}/fig_3_cost_comparison.pdf", bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 4. Coverage / Unmet Chart (Modern Stacked)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
width = 0.45

covered = [rb_units, ts_units]
unmet = [total_needed - rb_units, total_needed - ts_units]

ax.bar(labels, covered, width, label='Đã đáp ứng', color=COLOR_TS_ALT, zorder=3)
ax.bar(labels, unmet, width, bottom=covered, label='Chưa đáp ứng', color='#FFE66D', zorder=3)

ax.set_ylabel('Số lượng đơn vị (Units)', fontsize=13, labelpad=10)
ax.set_title('SO SÁNH TỶ LỆ ĐÁP ỨNG NHU CẦU', fontsize=16, pad=20, color='#2C3E50')
ax.set_ylim(0, total_needed * 1.2)
ax.tick_params(axis='x', labelsize=13, pad=10)

for i, (cov, unm) in enumerate(zip(covered, unmet)):
    cov_pct = cov / total_needed * 100
    ax.text(i, cov/2, f"{cov_pct:.0f}%\n({int(cov)})", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    if unm > 0:
        ax.text(i, cov + unm/2, f"{int(unm)}", ha='center', va='center', color='#333', fontweight='bold', fontsize=12)

ax.legend(loc='upper right', frameon=True, framealpha=1.0, edgecolor='#E0E0E0', fontsize=11)

plt.tight_layout()
plt.savefig(f"{out_dir}/fig_3_coverage_unmet.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{out_dir}/fig_3_coverage_unmet.pdf", bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 5. Network Graph (Premium Dark Mode)
# -----------------------------------------------------------------------------
def plot_premium_network(df_edges, filename, title, max_edges=None):
    agg_edges = df_edges.groupby(['from_store_id', 'to_store_id'])['units'].sum().reset_index()
    if max_edges and len(agg_edges) > max_edges:
        agg_edges = agg_edges.nlargest(max_edges, 'units')
        
    G = nx.DiGraph()
    for _, row in agg_edges.iterrows():
        u, v, w = int(row['from_store_id']), int(row['to_store_id']), row['units']
        G.add_edge(u, v, weight=w)
        
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=COLOR_DARK_BG)
    ax.set_facecolor(COLOR_DARK_BG)
    
    node_sizes = []
    for node in G.nodes():
        in_w = sum([d['weight'] for u, v, d in G.in_edges(node, data=True)])
        out_w = sum([d['weight'] for u, v, d in G.out_edges(node, data=True)])
        node_sizes.append(400 + (in_w + out_w)*3)

    edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    if edge_weights:
        max_w = max(edge_weights)
        edge_widths = [1 + 6 * (w / max_w) for w in edge_weights]
        # Use a colormap for edges based on weight
        edge_colors = [plt.cm.cool(w / max_w) for w in edge_weights]
    else:
        edge_widths = []
        edge_colors = []

    pos = nx.spring_layout(G, k=1.8, seed=42)
    
    # Draw glowing halo for nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=COLOR_NODE, node_size=[s*1.2 for s in node_sizes], alpha=0.3)
    # Draw core nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=COLOR_NODE, edgecolors='white', linewidths=1.5, node_size=node_sizes, alpha=0.9)
    
    # Draw curved edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors, arrowsize=18, alpha=0.8, 
                           connectionstyle="arc3,rad=0.15")
    
    # Node labels
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=11, font_weight='bold', font_color='black')
    
    ax.set_title(title.upper(), fontsize=18, fontweight='heavy', color='white', pad=20)
    ax.axis('off')
    
    import matplotlib.lines as mlines
    line = mlines.Line2D([], [], color='#D63031', linewidth=4, label='Tuyến điều chuyển (Độ dày & Màu sắc tỷ lệ với số lượng)')
    marker = mlines.Line2D([], [], color='white', marker='o', markerfacecolor=COLOR_NODE, markeredgecolor='white', 
                           markersize=12, label='Cửa hàng (Kích thước tỷ lệ với lưu lượng)')
    
    leg = ax.legend(handles=[line, marker], loc='lower right', frameon=True, facecolor='#2D3436', edgecolor='none', fontsize=11)
    for text in leg.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}.png", dpi=300, bbox_inches='tight', facecolor=COLOR_DARK_BG)
    plt.savefig(f"{out_dir}/{filename}.pdf", bbox_inches='tight', facecolor=COLOR_DARK_BG)
    plt.close()

plot_premium_network(df_ts, "fig_3_transfer_network", "Toàn bộ mạng lưới điều chuyển theo Transportation Simplex")
plot_premium_network(df_ts, "fig_3_transfer_network_top30", "30 tuyến điều chuyển lớn nhất theo Transportation Simplex", max_edges=30)

print("All assets regenerated successfully with PREMIUM aesthetics.")
