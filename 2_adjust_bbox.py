import os
from PIL import Image

# 读取cut_count.txt文件来获取各个cut区域的参考总数
def read_cut_counts(cut_count_file):
    cut_counts = {}
    with open(cut_count_file, 'r') as file:
        for line in file:
            cut_name, count = line.strip().split()
            cut_counts[cut_name] = int(count)
    return cut_counts


# 设定cut区域的参考总数
cut_count_file = 'info/cut_count.txt'  # 路径可根据实际情况修改
cut_counts = read_cut_counts(cut_count_file)

def adjust_bbox_ymin_yolo(image_dir, input_dir, output_dir, ymin_offset, region, info_dir):
    # 创建 box_count_summary.txt 文件并写入标题
    summary_file = os.path.join(info_dir, '修正_summary.txt')
    with open(summary_file, 'w') as summary:
        summary.write("Region\tDetected Boxes\tReference Count\tPercentage\n")

    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.jpg', '.png', '.tif')):
            image_path = os.path.join(image_dir, image_name)

            with Image.open(image_path) as img:
                image_width, image_height = img.size

            annotation_file = os.path.join(input_dir, image_name.split('.')[0] + '.txt')

            if not os.path.exists(annotation_file):
                print(f"标注文件 {annotation_file} 不存在，跳过该图像。")
                continue

            # 读取原标注文件并计算检测到的框的数量
            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            # 检查框的数量是否与参考数量匹配
            cut_name = image_name.split('.')[0]  # 假设图片名称就是cut名称
            if cut_name in cut_counts:
                detected_count = len(lines)
                reference_count = cut_counts[cut_name]
                # 计算百分比
                percentage = (detected_count / reference_count) * 100 if reference_count > 0 else 0
                # 输出框数量、参考数量、百分比到 summary 文件
                with open(summary_file, 'a') as summary:
                    summary.write(f"{cut_name}\t{detected_count}\t{reference_count}\t{percentage:.2f}%\n")

                if detected_count != reference_count:
                    print(f"警告：{cut_name} 区域的检测框数量不匹配！检测到: {detected_count}，参考数量: {reference_count}")
            else:
                print(f"警告：无法找到 {cut_name} 的参考数量信息")

            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, image_name.split('.')[0] + '.txt')

            with open(output_file, 'w') as f:
                for line in lines:
                    parts = line.split()

                    class_id = int(parts[0])  # 目标类别ID
                    center_x = float(parts[1])  # 中心点x坐标
                    center_y = float(parts[2])  # 中心点y坐标
                    width = float(parts[3])  # 目标框的宽度
                    height = float(parts[4])  # 目标框的高度

                    ymin = center_y - height / 2
                    ymax = center_y + height / 2

                    # 调整 ymin，保持 ymax 不变
                    new_ymin = ymin - ymin_offset / image_height  # 转换为归一化的ymin
                    new_height = ymax - new_ymin  # 新的高度

                    # 重新计算新的 center_y
                    new_center_y = new_ymin + new_height / 2  # 更新 center_y，确保中心位置不变

                    # 将新的信息写入到输出文件
                    f.write(f"{class_id} {center_x} {new_center_y} {width} {new_height}\n")

            print(f"图像 {image_name} 的标注文件已调整并保存至 {output_file}")

# 输入参数
region = 'Singapore'  # 输入区域名称
date = '2022_01_10'


input_dir = f"tank_file/{region}/{date}/1_original_tiff"  # 图像文件夹路径
txt_dir = f"tank_file/{region}/{date}/3_no_shadow_yolo_label/"  # 原始标注文件夹
output_dir = f"tank_file/{region}/{date}/4_shadow_yolo_label/"  # 输出标注文件夹
info_dir = f"tank_file/{region}/{date}/more_info/"  # 信息目录
extend_path = os.path.join(info_dir, 'extend_info.txt')
# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(info_dir, exist_ok=True)
# 调用函数进行调整
ymin_offset =25 # 要延伸的 ymin 值
# ymin_offset = 62  # 要延伸的 ymin 值

# 调用函数进行调整
adjust_bbox_ymin_yolo(input_dir, txt_dir, output_dir, ymin_offset, region, info_dir)

# 在 YOLO 标签文件夹中创建 classes.txt 文件
classes_txt_path = os.path.join(output_dir, 'classes.txt')
with open(classes_txt_path, 'w') as f:
    f.write('tank\n')

print("classes.txt 已保存到 YOLO 标签目录！extend_info.txt 已保存到 info 目录！")

# 保存 extend_info.txt 文件
with open(extend_path, 'w') as f:
    f.write(f"{ymin_offset}")
