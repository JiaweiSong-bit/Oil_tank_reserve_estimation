import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from pycpd import RigidRegistration
import os

def read_points_from_file(file_path):
    """
    从.txt文件中读取点集数据，文件格式为：ID x_coordinate y_coordinate

    参数：
    - file_path (str): .txt文件路径

    返回：
    - points (np.array): 点集数组，形状为(n x 2)
    """
    # 读取文件
    data = np.loadtxt(file_path)
    # 提取x, y坐标
    points = data[:, 1:3]
    return points


import numpy as np
from pycpd import RigidRegistration
from scipy.spatial.distance import cdist

def match_target_to_source(target, source, threshold=100):
    """
    函数：将目标点集（target）与源点集（source）进行匹配，并返回根据匹配结果排序的目标点顺序。

    参数：
    - target (np.array): 目标点集，二维数组（n x 2）。
    - source (np.array): 源点集，二维数组（n x 2）。
    - threshold (float): 匹配的距离阈值，默认为10。

    返回：
    - sorted_target_indices (list): 按照匹配结果排序的目标点索引列表。
    - TY (np.array): 变换后的源点集坐标。
    - closest_target_indices (np.array): 每个源点对应的最近目标点索引。
    """
    source=source-source[0]
    # 创建刚性配准对象
    reg = RigidRegistration(X=target, Y=source)

    # 执行配准并获取变换后的坐标
    TY_scaled, _ = reg.register()

    # 计算变换后的源点与目标点之间的欧氏距离
    distances = cdist(TY_scaled, target)

    # 为每个源点找到最接近的目标点
    closest_target_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    # 根据距离阈值筛选匹配点
    matching_points = [i if min_distances[i] <= threshold / 10.0 else None for i in range(len(source))]

    # 提取目标点的排序顺序
    sorted_target_indices = [closest_target_indices[i] for i in range(len(source)) if matching_points[i] is not None]

    return sorted_target_indices, TY_scaled, closest_target_indices




def visualize_matching(source, target, transformed_target, closest_target_indices, match_img_save_dir,image_name):
    """
    可视化匹配结果，将源点、目标点和变换后的目标点绘制在同一图上，并标出匹配的连接线。
    然后将图形保存到指定文件夹。

    参数：
    - source (np.array): 源点集，二维数组（n x 2）。
    - target (np.array): 目标点集，二维数组（n x 2）。
    - transformed_target (np.array): 变换后的目标点集，二维数组（n x 2）。
    - closest_target_indices (list): 源点匹配到的目标点索引。
    - match_img_save_dir (str): 保存匹配结果图像的文件夹路径。
    """
    plt.figure(figsize=(10, 10))

    # 绘制源点和目标点
    plt.scatter(source[:, 0], source[:, 1], color='blue', label='Source Points', s=50)
    plt.scatter(target[:, 0], target[:, 1], color='green', label='Target Points', s=50)
    plt.scatter(transformed_target[:, 0], transformed_target[:, 1], color='red', label='Transformed Target Points', s=50)

    # 绘制匹配的连接线
    for i in range(len(source)):
        plt.plot([source[i, 0], target[closest_target_indices[i], 0]],
                 [source[i, 1], target[closest_target_indices[i], 1]], color='gray', linestyle='--')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Matching of Source and Target Points')
    plt.grid(True)
    plt.axis('equal')

    # 保存图形到文件夹
    os.makedirs(match_img_save_dir, exist_ok=True)
    output_file_path = f"{match_img_save_dir}/{image_name}.png"
    plt.savefig(output_file_path)
    # 关闭当前图形
    plt.close()
    # print(f"图形已保存到 {output_file_path}")
