import os
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from CPD import match_target_to_source,visualize_matching

def rotate_point(x, y, angle_deg):
    # 计算旋转角度的弧度值
    angle_rad = math.radians(angle_deg)

    # 旋转矩阵中的常数
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    # 逆时针旋转的公式
    x_rot = x * cos_angle - y * sin_angle
    y_rot = x * sin_angle + y * cos_angle

    return x_rot, y_rot


def read_rotation_angle(input_image_dir):
    # 从rotation_angle.txt文件中读取旋转角度
    rotation_file = os.path.join(input_image_dir, 'rotation_angle.txt')
    if not os.path.exists(rotation_file):
        raise FileNotFoundError(f"{rotation_file} not found.")

    with open(rotation_file, 'r') as f:
        line = f.readline().strip()
        if line.startswith("Rotation angle:"):
            angle_str = line.split(":")[1].strip()
            print(f"已读取到旋转角度{angle_str}")
            return float(angle_str)
        else:
            raise ValueError(f"Invalid format in {rotation_file}")

def read_extend(input_image_dir):
    # 从extend_info.txt文件中读取旋转角度
    extend_file = os.path.join(input_image_dir, 'extend_info.txt')
    if not os.path.exists(extend_file):
        raise FileNotFoundError(f"{extend_file} not found.")

    with open(extend_file, 'r') as f:
        line = f.readline().strip()
        extend_str = line
        print(f"已读取到延伸长度{extend_str}")
        return float(extend_str)

def read_points_from_file(file_path):
    data = np.loadtxt(file_path)
    # 提取x, y坐标
    points = data[:, 1:3]
    return points

def match_and_label_bboxes_sorted(a_dir, b_dir, output_dir, check_dir, rotation_angle, center_output_dir,region,date):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)
    os.makedirs(center_output_dir, exist_ok=True)  # 创建保存中心点的文件夹

    # 加载字体并设置字体大小
    try:
        font = ImageFont.truetype("arial.ttf", 50)  # 设置字体大小为30，可以根据需要调整
    except IOError:
        font = ImageFont.load_default()  # 如果没有找到该字体，使用默认字体

    for annotation_file in os.listdir(a_dir):
        if annotation_file.endswith('.txt'):
            image_name = annotation_file.split('.')[0]
            if image_name == 'classes':
                continue
            annotation_path_a = os.path.join(a_dir, image_name + '.txt')
            annotation_path_b = os.path.join(b_dir, image_name + '.txt')

            if not os.path.exists(annotation_path_b):
                print(f"图像 {image_name} 的调整后的标注文件不存在，跳过该图像。")
                continue

            with open(annotation_path_a, 'r') as f_a, open(annotation_path_b, 'r') as f_b:
                lines_a = f_a.readlines()
                lines_b = f_b.readlines()

            if len(lines_a) != len(lines_b):
                print(f"图像 {image_name} 的原始框和调整后的框数量不匹配，跳过该图像。")
                continue

            # 将框的坐标转换为中心坐标并进行排序
            boxes_a = []
            boxes_b = []
            temp_a = []
            temp_count = 0

            image_path = os.path.join(input_image_dir, image_name + '.tif')
            img = Image.open(image_path)

            for line_a in lines_a:
                parts_a = line_a.split()
                temp_a.append(parts_a)
                center_x_a = float(parts_a[1])* img.width
                center_y_a = float(parts_a[2])* img.height
                # 逆时针旋转角度（来自rotation_angle.txt）
                rotated_x_a, rotated_y_a = rotate_point(center_x_a, center_y_a, rotation_angle)
                boxes_a.append((rotated_x_a, rotated_y_a, line_a))  # 记录旋转后的坐标和原始框内容

            source_file_path = 'info/分区油罐坐标参考/'+region+'/'+image_name+'.txt'  # 以2024_12_07为参考基准不要改动
            # 读取源点和目标点数据
            source = read_points_from_file(source_file_path)
            # 将boxes_a中的旋转后的坐标作为目标点集
            target = np.array([box[0:2] for box in boxes_a])
            # 获取匹配结果
            sorted_target_indices, transformed_target, closest_target_indices = match_target_to_source(target, source)

            print(sorted_target_indices)
            match_img_save_dir=check_dir+'/match/'
            visualize_matching(source, target, transformed_target, closest_target_indices,match_img_save_dir,image_name)

            # 根据sorted_target_indices对boxes_a和boxes_b进行排序
            boxes_a_sorted = [boxes_a[i] for i in sorted_target_indices]

            for line_b in lines_b:
                parts_b = temp_a[temp_count]
                center_x_b = float(parts_b[1])* img.width
                center_y_b = float(parts_b[2])* img.height
                # 逆时针旋转角度（来自rotation_angle.txt）
                rotated_x_b, rotated_y_b = rotate_point(center_x_b, center_y_b, rotation_angle)
                boxes_b.append((rotated_x_b, rotated_y_b, line_b))  # 记录旋转后的坐标和调整框内容
                temp_count += 1

            boxes_b_sorted = [boxes_b[i] for i in sorted_target_indices]
            # 输出匹配后的框
            output_file = os.path.join(output_dir, image_name + '_matched.txt')

            target_id = 1  # 从目标ID 1开始
            with open(output_file, 'w') as f_out:
                # 可视化框到原图像
                draw = ImageDraw.Draw(img)

                # 创建新的文件来保存boxes_a的中心点坐标
                center_output_file = os.path.join(center_output_dir, image_name + '.txt')
                with open(center_output_file, 'w') as f_center:
                    count=0
                    for box_a, box_b in zip(boxes_a_sorted, boxes_b_sorted):
                        # 处理原始框
                        count+=1
                        parts_a = box_a[2].split()
                        center_x_a = float(parts_a[1])
                        center_y_a = float(parts_a[2])

                        # 计算实际坐标（像素值）
                        actual_center_x_a, actual_center_y_a = center_x_a * img.width, center_y_a * img.height

                        # 记录转换后的中心点坐标到txt文件
                        f_center.write(f"{count} {actual_center_x_a} {actual_center_y_a}\n")

                        # 处理原始框（a_dir中的框）
                        class_id_a = int(parts_a[0])
                        width_a = float(parts_a[3])
                        height_a = float(parts_a[4])

                        # 画原始框（a_dir中的框）
                        xmin_a, ymin_a, xmax_a, ymax_a = convert_yolo_to_bbox(center_x_a, center_y_a, width_a, height_a,
                                                                              img.width, img.height)
                        draw.rectangle([xmin_a, ymin_a, xmax_a, ymax_a], outline="red", width=2)

                        # 标注目标ID到框内
                        text_position = (xmin_a + 15, ymin_a + 15)  # 文本位置
                        draw.text(text_position, str(target_id), fill="blue", font=font)  # 使用指定的字体大小

                        # 在中心点绘制一个小圆点
                        center_point_radius = 4  # 小圆点的半径
                        center_point_position = (center_x_a * img.width, center_y_a * img.height)
                        draw.ellipse(
                            [center_point_position[0] - center_point_radius, center_point_position[1] - center_point_radius,
                             center_point_position[0] + center_point_radius,
                             center_point_position[1] + center_point_radius],
                            fill="black")  # 绿色的小圆点

                        # 处理调整后的框
                        parts_b = box_b[2].split()
                        class_id_b = int(parts_b[0])
                        center_x_b = float(parts_b[1])
                        center_y_b = float(parts_b[2])
                        width_b = float(parts_b[3])
                        height_b = float(parts_b[4])

                        f_out.write(f"{target_id} {class_id_a} {center_x_a} {center_y_a} {width_a} {height_a} 0\n")
                        f_out.write(f"{target_id} {class_id_b} {center_x_b} {center_y_b} {width_b} {height_b} 1\n")

                        target_id += 1

                # 保存带有可视化框的图像到check文件夹
                check_image_path = os.path.join(check_dir, image_name + '.tif')
                img.save(check_image_path)

    print("标签和框的匹配处理完成。")


def convert_yolo_to_bbox(x_center, y_center, width, height, image_width, image_height):

    xmin = int((x_center - width / 2) * image_width)
    ymin = int((y_center - height / 2) * image_height)
    xmax = int((x_center + width / 2) * image_width)
    ymax = int((y_center + height / 2) * image_height)
    return xmin, ymin, xmax, ymax

def calculate_overlap(box1, box2):
    """
    计算两个边界框的重叠区域。
    """
    overlap_x_min = max(box1[0], box2[0])
    overlap_x_max = min(box1[2], box2[2])
    overlap_y_min = max(box1[1], box2[1])
    overlap_y_max = min(box1[3], box2[3])

    if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:  # 有重叠
        overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
        overlap_center_x = (overlap_x_min + overlap_x_max) / 2
        return overlap_area, overlap_center_x
    return 0, None

def crop_image(image_path, yolo_txt_path, noshadow_output_dir, shadow_output_dir, image_width, image_height):
    """
    根据YOLO格式的标注裁剪图像
    """
    img = Image.open(image_path)

    with open(yolo_txt_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    for line in lines:
        parts = line.strip().split()
        target_id = int(parts[0])
        class_id = int(parts[1])
        x_center = float(parts[2])
        y_center = float(parts[3])
        width = float(parts[4])
        height = float(parts[5])
        with_shadow = int(parts[6])
        xmin, ymin, xmax, ymax = convert_yolo_to_bbox(x_center, y_center, width, height, image_width, image_height)
        bboxes.append((target_id, xmin, ymin, xmax, ymax, with_shadow))

    for i, (target_id, xmin, ymin, xmax, ymax, with_shadow) in enumerate(bboxes):
        cropped_img = img.crop((xmin, ymin, xmax, ymax))

        marked_name = f"_N"  # 默认没有重叠为_N
        overlap_count = 0

        if with_shadow == 1:
            for j, (other_id, xmin2, ymin2, xmax2, ymax2, with_shadow2) in enumerate(bboxes):
                if i != j and ymin2 < ymin and with_shadow2 == 0:  # 上方无阴影框
                    overlap_area, overlap_center_x = calculate_overlap((xmin, ymin, xmax, ymax),
                                                                       (xmin2, ymin2, xmax2, ymax2))
                    if overlap_area > 5:  # 重叠面积达到5像素
                        overlap_count += 1
                        if overlap_count == 1:
                            if overlap_center_x < (xmin + xmax) / 2:
                                marked_name = f"_L"
                            else:
                                marked_name = f"_R"
                        elif overlap_count >= 2:  # 上方重叠2个检测框
                            marked_name = f"_N"
                            break
        # 计算裁剪图像的中心点坐标
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        # 保存裁剪图像
        if with_shadow == 0:
            output_image_path = os.path.join(noshadow_output_dir,
                                             f"{os.path.splitext(os.path.basename(image_path))[0]}_{target_id}.tif")
        else:
            output_image_path = os.path.join(shadow_output_dir,
                                             f"{os.path.splitext(os.path.basename(image_path))[0]}_{target_id}{marked_name}.tif")

        cropped_img.save(output_image_path, format='TIFF')
        print(f"Cropped image saved: {output_image_path}")



region = 'Chiba'  # 输入区域名称
date = '2023_10_12'

# 输入图像和标注文件的目录路径
input_image_dir = "tank_file/" + region + "/" + date + "/" + "1_original_tiff"  # 图像文件所在目录路径
a_dir = "tank_file/" + region + "/" + date + "/" + "3_no_shadow_yolo_label/"  # 原始标注文件所在目录路径
b_dir = "tank_file/" + region + "/" + date + "/" + "4_shadow_yolo_label/"  # 调整后的标注文件所在目录路径
input_txt_dir = "tank_file/" + region + "/" + date + "/" + "5_match/"  # 输出匹配后的标注文件的目录路径
noshadow_output_dir = "tank_file/" + region + "/" + date + "/" + "no_shadow_split_tank"  # 裁剪后的图像保存路径
shadow_output_dir = "tank_file/" + region + "/" + date + "/" + "shadow_split_tank"
center_dir = "tank_file/" + region + "/" + date + "/" + "6_center"  # 保存中心点坐标的txt文件路径
check_dir = "tank_file/" + region + "/" + date + "/" + "check"  # 保存检查图像的目录
info_dir="tank_file/" + region + "/" + date + "/" + "more_info"  # 保存检查图像的目录

# 读取rotation_angle.txt文件中的旋转角度
# rotation_angle = -read_rotation_angle(input_image_dir)
# rotation_angle = 0
extend=read_extend(info_dir)
# 调用函数进行调整并标号

match_and_label_bboxes_sorted(a_dir, b_dir, input_txt_dir, check_dir, rotation_angle,center_dir,region,date)

os.makedirs(noshadow_output_dir, exist_ok=True)
os.makedirs(shadow_output_dir, exist_ok=True)


for image_name in os.listdir(input_image_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png', '.tif')):
        image_path = os.path.join(input_image_dir, image_name)
        yolo_txt_path = os.path.join(input_txt_dir, os.path.splitext(image_name)[0] + '_matched' + '.txt')
        print(yolo_txt_path)
        if os.path.exists(yolo_txt_path):
            img = Image.open(image_path)
            image_width, image_height = img.size
            crop_image(image_path, yolo_txt_path, noshadow_output_dir, shadow_output_dir, image_width, image_height)

print("complete")

#--------------------------------------------代码自检匹配结果---------------------------------------------
def check_image_matching(noshadow_output_dir, shadow_output_dir,extend):
    # 获取 no_shadow_split_tank 文件夹中的所有图像文件
    noshadow_images = [f for f in os.listdir(noshadow_output_dir) if f.endswith('.tif')]

    # 用于记录是否有匹配失败的标志
    all_matched = True

    # 遍历每个 no_shadow_split_tank 图像文件
    for noshadow_image in noshadow_images:
        # 提取文件名（不包括扩展名）
        image_name = os.path.splitext(noshadow_image)[0]

        # 生成对应的 shadow_split_tank 文件名（可能有 "_N" 或 "_L" 或 "_R" 后缀）
        shadow_image_1 = image_name + "_N.tif"
        shadow_image_2 = image_name + "_L.tif"
        shadow_image_3 = image_name + "_R.tif"

        # 检查 shadow_split_tank 中是否存在相应的图像文件
        shadow_image_path = None
        if os.path.exists(os.path.join(shadow_output_dir, shadow_image_1)):
            shadow_image_path = os.path.join(shadow_output_dir, shadow_image_1)
        elif os.path.exists(os.path.join(shadow_output_dir, shadow_image_2)):
            shadow_image_path = os.path.join(shadow_output_dir, shadow_image_2)
        elif os.path.exists(os.path.join(shadow_output_dir, shadow_image_3)):
            shadow_image_path = os.path.join(shadow_output_dir, shadow_image_3)

        # 如果找不到匹配的图像，跳过该图像
        if shadow_image_path is None:
            print(f"警告: 没有找到与 {noshadow_image} 匹配的阴影图像。")
            all_matched = False
            continue

        # 加载 no_shadow_split_tank 和 shadow_split_tank 中的图像
        noshadow_img = Image.open(os.path.join(noshadow_output_dir, noshadow_image))
        shadow_img = Image.open(shadow_image_path)

        # 检查图像的宽度和高度是否符合要求
        noshadow_width, noshadow_height = noshadow_img.size
        shadow_width, shadow_height = shadow_img.size

        # 匹配条件：no_shadow图像的宽度和shadow图像的宽度相同，且高度比shadow图像少extend
        if noshadow_width == shadow_width and abs(shadow_height - extend- noshadow_height)<=1:
            continue  # 匹配成功，不输出任何信息
        else:
            print(f"匹配失败: {noshadow_image} 和 {shadow_image_path}，尺寸不匹配！")
            all_matched = False

    # 最后检查是否所有图像都成功匹配
    if all_matched:
        print("-----------代码自检匹配结果-----------\n所有图像匹配成功！")
    else:
        print("-----------代码自检匹配结果-----------\n部分图像匹配失败，请检查输出信息。")


# 调用函数来检查匹配
check_image_matching(noshadow_output_dir, shadow_output_dir,extend)
