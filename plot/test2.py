import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from matplotlib.patches import Ellipse
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch

data_folder_path = './test' 

all_data = []

print(f"正在尝试从文件夹 '{data_folder_path}' 加载所有 .jsonl 文件...")
found_files = False
if os.path.exists(data_folder_path):
    for file_name in os.listdir(data_folder_path):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(data_folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_data.append(json.loads(line))
                print(f"已加载文件: {file_name}")
                found_files = True
            except json.JSONDecodeError as e:
                print(f"警告: 文件 '{file_name}' 解析失败 (JSON 格式错误): {e}")
            except Exception as e:
                print(f"警告: 读取文件 '{file_name}' 时发生未知错误: {e}")
else:
    print(f"错误：未找到数据文件夹 '{data_folder_path}'。请创建此文件夹并将您的 .jsonl 文件放入其中。")
    raise FileNotFoundError(f"数据文件夹 '{data_folder_path}' 不存在。")

if not found_files:
    print(f"在 '{data_folder_path}' 文件夹中未找到任何 .jsonl 文件。")
    raise Exception("未找到数据文件。请确保文件夹中包含 .jsonl 文件。")
print(f"所有 .jsonl 文件加载完成。总共 {len(all_data)} 条记录。")


df = pd.DataFrame(all_data)


df_gptfuzz_1 = df[df['gptfuzz'] == 1].copy()
print(f"原始数据条数: {len(df)}")
print(f"gptfuzz=1 的数据条数: {len(df_gptfuzz_1)}")

if df_gptfuzz_1.empty:
    print("筛选后没有 gptfuzz=1 的数据，无法进行后续分析和绘图。请检查您的数据。")
    raise Exception("没有 gptfuzz=1 的数据。")
else:
    print("数据加载和筛选完成。")
    print("前5行筛选后的数据:")
    print(df_gptfuzz_1.head())
    print("\n'method' 列的分布:")
    print(df_gptfuzz_1['method'].value_counts())
    print("\n'success' 列的分布:")
    print(df_gptfuzz_1['success'].value_counts())


try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to("cuda:0")
except Exception as e:
    print(f"加载SentenceTransformer模型失败: {e}")
    print("请检查网络连接或手动下载模型: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
    raise Exception(f"加载SentenceTransformer模型失败: {e}")

print("正在生成 embeddings...")
embeddings = model.encode(df_gptfuzz_1['response'].tolist(), show_progress_bar=True)
print("Embeddings 生成完成。")

print("正在进行 UMAP 降维...")
reducer = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)
df_gptfuzz_1[['UMAP-1', 'UMAP-2']] = umap_embeddings
print("UMAP 降维完成。")


fig_width = 8 
fig_height = 7 
plt.figure(figsize=(fig_width, fig_height))


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'


sns.set_style("white") 
plt.grid(False) 

method_color_map = {
    'baseline': 'dodgerblue',
    'uja': 'darkorange'
}


marker_style_map = {
    True: 'o',
}


alpha_map = {
    True: 0.7,
}


lw_map = {
    True: 1.5, 
}

legend_handles = []


df_gptfuzz_1_success = df_gptfuzz_1[df_gptfuzz_1['success'] == True].copy()


for method in sorted(df_gptfuzz_1_success['method'].unique()):
    subset = df_gptfuzz_1_success[df_gptfuzz_1_success['method'] == method].copy()

    if not subset.empty:
        label = f'{method}'
        plt.scatter(subset['UMAP-1'], subset['UMAP-2'],
                    color=method_color_map[method],
                    marker=marker_style_map[True], 
                    alpha=alpha_map[True],       
                    s=70,                        
                    edgecolor='black',           
                    linewidth=lw_map[True]       
                    )
        legend_handles.append(plt.Line2D([0], [0], marker=marker_style_map[True], color='w',
                                         markerfacecolor=method_color_map[method], markersize=10,
                                         label=label, markeredgecolor='black',
                                         markeredgewidth=lw_map[True]))
        if len(subset) > 1:
            cov = np.cov(subset[['UMAP-1', 'UMAP-2']].T)
            mean = subset[['UMAP-1', 'UMAP-2']].mean().values

            epsilon = 1e-6
            cov += np.eye(cov.shape[0]) * epsilon

            λ, v = np.linalg.eig(cov)
            λ = np.sqrt(λ)
            angle = np.degrees(np.arctan2(*v[:, 0][::-1]))


            width_factor = 2.5 
            height_factor = 2.5 

            linestyle = '-' 
            alpha_ellipse = 0.6 
            lw_ellipse = 2 

            ellipse = Ellipse(xy=mean, width=λ[0]*width_factor, height=λ[1]*height_factor, angle=angle,
                              edgecolor=method_color_map[method], fc='None', linestyle=linestyle,
                              lw=lw_ellipse, alpha=alpha_ellipse)
            plt.gca().add_patch(ellipse)
        elif len(subset) == 1:
            print(f"警告: {method} 组只有一个成功数据点，无法绘制椭圆。")


plt.legend(handles=legend_handles, bbox_to_anchor=(1.0, 1), loc="upper left", fontsize=10, frameon=False)


plt.tight_layout(rect=[0, 0, 0.85, 1]) 

output_filename = os.path.join(data_folder_path, "umap_visualization_gptfuzz1.png")
plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='png')
print(f"图像已保存至: {output_filename}")

plt.show()