import os
from PIL import Image
from mmdet.apis import init_detector, inference_detector


# ===============================
# 读取cut_count.txt
# ===============================
def read_cut_counts(cut_count_file):
    cut_counts = {}
    with open(cut_count_file, 'r') as file:
        for line in file:
            cut_name, count = line.strip().split()
            cut_counts[cut_name] = int(count)
    return cut_counts


cut_count_file = 'info/cut_count.txt'
cut_counts = read_cut_counts(cut_count_file)


# ===============================
# 转换为YOLO格式
# ===============================
def convert_to_yolo_format(input_file, output_file, image_path):

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:

        for line in infile:

            parts = line.strip().split()

            class_id = int(parts[0])
            xmin = float(parts[2])
            ymin = float(parts[3])
            xmax = float(parts[4])
            ymax = float(parts[5])

            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            outfile.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# ===============================
# 批量转换YOLO
# ===============================
def process_all_images_in_directory(input_dir, txt_dir, output_dir):

    total_boxes = 0

    cut_counts_detected = {cut_name: 0 for cut_name in cut_counts.keys()}

    for image_name in os.listdir(input_dir):

        if not image_name.lower().endswith('.tif'):
            continue

        image_path = os.path.join(input_dir, image_name)

        txt_file = os.path.splitext(image_name)[0] + '.txt'
        input_file = os.path.join(txt_dir, txt_file)
        output_file = os.path.join(output_dir, txt_file)

        if os.path.exists(input_file):

            convert_to_yolo_format(input_file, output_file, image_path)

            with open(output_file, 'r') as f:
                num_boxes = sum(1 for _ in f)

            total_boxes += num_boxes

            for cut_name in cut_counts.keys():
                if cut_name.lower() in image_name.lower():
                    cut_counts_detected[cut_name] += num_boxes

        else:
            print(f"Warning: {input_file} does not exist!")

    return total_boxes, cut_counts_detected


# ===============================
# 模型初始化
# ===============================
config_file = "configs/cascade_swin_oil.py"
checkpoint_file = "work_dir/cascade_rcnn_swin_oil/epoch_29.pth"

model = init_detector(config_file, checkpoint_file, device='cuda:0')

classes = '0'


# ===============================
# 主批处理逻辑
# ===============================
base_input_dir = "sp"

for region in os.listdir(base_input_dir):

    region_dir = os.path.join(base_input_dir, region)

    if not os.path.isdir(region_dir):
        continue

    print(f"\n========== Processing Region: {region} ==========")

    for date in os.listdir(region_dir):

        date_dir = os.path.join(region_dir, date)

        if not os.path.isdir(date_dir):
            continue

        input_dir = os.path.join(date_dir, "1_original_tiff")

        if not os.path.exists(input_dir):
            print(f"Skip {region}/{date}, no 1_original_tiff folder")
            continue

        print(f"\n---- Processing {region}/{date} ----")

        mid_dir = os.path.join(date_dir, "2_model_vison_label")
        output_dir = os.path.join(date_dir, "3_no_shadow_yolo_label")
        info_dir = os.path.join(date_dir, "more_info")

        os.makedirs(mid_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(info_dir, exist_ok=True)

        cut_counts_detected = {cut_name: 0 for cut_name in cut_counts.keys()}
        total_detection_boxes = 0

        summary_path = os.path.join(info_dir, "检测_summary.txt")

        with open(summary_path, 'w') as summary_file:

            for image_name in os.listdir(input_dir):

                if not image_name.lower().endswith('.tif'):
                    continue

                image_path = os.path.join(input_dir, image_name)

                result = inference_detector(model, image_path)

                output_txt = os.path.join(mid_dir, f'{os.path.splitext(image_name)[0]}.txt')

                with open(output_txt, 'w') as f:

                    if isinstance(result, list):

                        for i, class_result in enumerate(result):

                            if class_result.size == 0:
                                continue

                            for box in class_result:

                                class_name = classes[i]
                                confidence = box[4]
                                xmin, ymin, xmax, ymax = box[:4]

                                f.write(f"{class_name} {confidence:.4f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}\n")

                    else:

                        pred_instances = result.pred_instances

                        if pred_instances is not None:

                            boxes = pred_instances.bboxes.cpu().numpy()
                            scores = pred_instances.scores.cpu().numpy()

                            for i in range(len(boxes)):

                                xmin, ymin, xmax, ymax = boxes[i]
                                confidence = scores[i]

                                f.write(f"0 {confidence:.4f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}\n")

                with open(output_txt, 'r') as f:
                    num_boxes = sum(1 for _ in f)

                total_detection_boxes += num_boxes

                cut_name = os.path.splitext(image_name)[0]
                cut_counts_detected[cut_name] = num_boxes

                print(f"Processed {region}/{date}/{image_name}, detected {num_boxes} boxes")

            summary_file.write(f"--- {region} {date} Detection Summary ---\n")

            for cut_name, detected_count in cut_counts_detected.items():

                if cut_name in cut_counts:

                    reference_count = cut_counts[cut_name]
                    percentage = (detected_count / reference_count) * 100

                    summary_file.write(
                        f"{cut_name} total boxes: {detected_count} ({percentage:.2f}%)\n"
                    )

            percentage_overall = (total_detection_boxes / sum(cut_counts.values())) * 100

            summary_file.write(
                f"Total boxes in {date}: {total_detection_boxes} ({percentage_overall:.2f}%)\n"
            )

        process_all_images_in_directory(input_dir, mid_dir, output_dir)

        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            f.write('tank\n')

        print(f"YOLO labels generated for {region}/{date}")

print("\n所有区域所有日期处理完成！")