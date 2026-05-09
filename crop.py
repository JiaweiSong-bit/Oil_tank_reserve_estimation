import os
from PIL import Image

# ==============================
# YOLO 转 bbox
# ==============================
def convert_yolo_to_bbox(x_center, y_center, width, height, image_width, image_height):
    xmin = int((x_center - width / 2) * image_width)
    ymin = int((y_center - height / 2) * image_height)
    xmax = int((x_center + width / 2) * image_width)
    ymax = int((y_center + height / 2) * image_height)
    return xmin, ymin, xmax, ymax


# ==============================
# 计算重叠
# ==============================
def calculate_overlap(box1, box2):
    overlap_x_min = max(box1[0], box2[0])
    overlap_x_max = min(box1[2], box2[2])
    overlap_y_min = max(box1[1], box2[1])
    overlap_y_max = min(box1[3], box2[3])

    if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
        overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
        overlap_center_x = (overlap_x_min + overlap_x_max) / 2
        return overlap_area, overlap_center_x

    return 0, None


# ==============================
# 裁剪函数
# ==============================
def crop_from_yolo(image_path, yolo_txt_path, output_dir, check_overlap=False):
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path)
    image_width, image_height = img.size

    with open(yolo_txt_path, 'r') as f:
        lines = f.readlines()

    bboxes = []

    for idx, line in enumerate(lines):
        parts = line.strip().split()

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        xmin, ymin, xmax, ymax = convert_yolo_to_bbox(
            x_center, y_center, width, height,
            image_width, image_height
        )

        bboxes.append((idx + 1, xmin, ymin, xmax, ymax))

    # ==============================
    # 开始裁剪
    # ==============================
    for i, (target_id, xmin, ymin, xmax, ymax) in enumerate(bboxes):

        cropped_img = img.crop((xmin, ymin, xmax, ymax))

        # 默认不加后缀
        suffix = ""

        # 只有 shadow 才判断重叠
        if check_overlap:
            suffix = "_N"
            overlap_count = 0

            for j, (_, xmin2, ymin2, xmax2, ymax2) in enumerate(bboxes):
                if i != j and ymin2 < ymin:
                    overlap_area, overlap_center_x = calculate_overlap(
                        (xmin, ymin, xmax, ymax),
                        (xmin2, ymin2, xmax2, ymax2)
                    )

                    if overlap_area > 5:
                        overlap_count += 1

                        if overlap_count == 1:
                            if overlap_center_x < (xmin + xmax) / 2:
                                suffix = "_L"
                            else:
                                suffix = "_R"
                        elif overlap_count >= 2:
                            suffix = "_N"
                            break

        save_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{target_id}{suffix}.tif"
        save_path = os.path.join(output_dir, save_name)

        cropped_img.save(save_path, format='TIFF')
        print("Saved:", save_path)


# ==============================
# 主程序
# ==============================
# 输入参数
region = 'Singapore'  # 输入区域名称
date = '2022_01_10'

input_image_dir = f"tank_file/{region}/{date}/1_original_tiff"
a_dir = f"tank_file/{region}/{date}/3_no_shadow_yolo_label"
b_dir = f"tank_file/{region}/{date}/4_shadow_yolo_label"

noshadow_output_dir = f"tank_file/{region}/{date}/no_shadow_split_tank"
shadow_output_dir = f"tank_file/{region}/{date}/shadow_split_tank"

os.makedirs(noshadow_output_dir, exist_ok=True)
os.makedirs(shadow_output_dir, exist_ok=True)

# ==============================
# 遍历图像
# ==============================
for image_name in os.listdir(input_image_dir):
    if image_name.endswith('.tif'):

        image_path = os.path.join(input_image_dir, image_name)
        txt_name = os.path.splitext(image_name)[0] + ".txt"

        txt_a = os.path.join(a_dir, txt_name)
        txt_b = os.path.join(b_dir, txt_name)

        # --- 直接裁 no_shadow ---
        if os.path.exists(txt_a):
            print("裁剪 no_shadow:", image_name)
            crop_from_yolo(image_path, txt_a, noshadow_output_dir, check_overlap=False)

        # --- 直接裁 shadow ---
        if os.path.exists(txt_b):
            print("裁剪 shadow:", image_name)
            crop_from_yolo(image_path, txt_b, shadow_output_dir, check_overlap=True)

print("complete")