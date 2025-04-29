import cv2
import numpy as np
import os

def generate_points_from_binary(binary_img, step=10):
    """
    Tạo các điểm tọa độ cách đều nhau từ ảnh nhị phân.

    Args:
        binary_img: Ảnh nhị phân (255 cho lane, 0 cho background).
        step: Khoảng cách giữa các điểm theo trục y (mặc định 10 pixel).

    Returns:
        points_per_object: Danh sách các danh sách điểm [(x, y), ...] cho mỗi làn đường.
    """
    # Tìm các thành phần liên thông
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_img.astype(np.uint8), connectivity=8)

    points_per_object = []
    for label in range(1, num_labels):  # Bỏ qua label 0 (background)
        # Lấy các pixel thuộc thành phần liên thông
        mask = (labels_im == label).astype(np.uint8)
        y_coords, x_coords = np.where(mask)

        if len(y_coords) < 10:  # Bỏ qua các thành phần quá nhỏ
            continue

        # Tạo điểm cách đều theo trục y
        y_min, y_max = y_coords.min(), y_coords.max()
        y_start = round(y_min / step) * step
        y_end = round(y_max / step) * step
        y_range = np.arange(y_start, y_end + step, step)

        points = []
        for y in y_range:
            # Lấy tất cả x tại y hiện tại
            x_vals = x_coords[y_coords == y]
            if x_vals.size > 0:
                # Tính x trung bình (hoặc chọn x đầu tiên nếu cần)
                x = np.mean(x_vals).item()
                points.append((x, y))

        if points:
            points_per_object.append(points)

    return points_per_object

def convert_binary_to_points(input_dir, output_dir, step=10):
    """
    Chuyển đổi tất cả ảnh nhị phân trong thư mục đầu vào sang định dạng point.

    Args:
        input_dir: Thư mục chứa ảnh nhị phân (ví dụ: /root/bdd100k/ll_seg_annotations/train/).
        output_dir: Thư mục lưu file .lines.txt (ví dụ: /root/bdd100k/lane_points/train/).
        step: Khoảng cách giữa các điểm (mặc định 10 pixel).
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lấy danh sách file ảnh nhị phân
    valid_extensions = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    for file in files:
        # Đọc ảnh nhị phân
        binary_img = cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_GRAYSCALE)
        if binary_img is None:
            print(f"Không thể đọc ảnh: {file}")
            continue

        # Tạo điểm từ ảnh nhị phân
        points_per_object = generate_points_from_binary(binary_img, step=step)

        # Lưu điểm vào file .lines.txt
        base_name = os.path.splitext(file)[0]
        txt_path = os.path.join(output_dir, f"{base_name}.lines.txt")
        try:
            with open(txt_path, "w") as f:
                for points in points_per_object:
                    # Sắp xếp điểm theo y (từ trên xuống dưới)
                    sorted_points = sorted(points, key=lambda p: p[1], reverse=True)
                    # Format: x1 y1 x2 y2 ...
                    line_str = " ".join([f"{pt[0]:.4f} {pt[1]}" for pt in sorted_points])
                    f.write(line_str + "\n")
            print(f"Đã lưu điểm vào: {txt_path}")
        except Exception as e:
            print(f"Lỗi khi lưu {txt_path}: {str(e)}")

if __name__ == "__main__":
    # Thư mục đầu vào và đầu ra
    input_dir = "/root/bdd100k/ll_seg_annotations/train/"  # Thư mục chứa ảnh nhị phân
    output_dir = "/root/bdd100k/lane_points/train/"       # Thư mục lưu file .lines.txt

    # Chạy chuyển đổi
    convert_binary_to_points(input_dir, output_dir, step=10)

    # Tương tự cho tập val nếu cần
    input_dir_val = "/root/bdd100k/ll_seg_annotations/val/"
    output_dir_val = "/root/bdd100k/lane_points/val/"
    convert_binary_to_points(input_dir_val, output_dir_val, step=10)