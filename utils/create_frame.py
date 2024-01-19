import cv2
import os
import glob

def video_to_frames(input_folder, output_folder, target_fps=1):
    # Lấy danh sách tất cả các file video trong thư mục đầu vào
    video_files = glob.glob(os.path.join(input_folder, '*.mp4'))
    print(video_files)
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in video_files:
        # Mở video
        cap = cv2.VideoCapture(video_file)

        # Đảm bảo video mở thành công
        if not cap.isOpened():
            print(f"Không thể mở video {video_file}.")
            continue

        # Lấy tên video để sử dụng làm tiền tố cho tên frame
        video_name = os.path.splitext(os.path.basename(video_file))[0]

        # Lấy fps của video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Tính toán số frame cần skip để đạt được target_fps
        skip_frames = int(round(fps / target_fps))

        # Đọc từng frame và lưu vào thư mục đầu ra
        frame_count = 0
        while True:
            ret, frame = cap.read()

            # Kiểm tra nếu đọc hết video
            if not ret:
                break

            # Skip frames để đạt được target_fps
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            # Lưu frame vào thư mục đầu ra
            frame_filename = f"{output_folder}/{video_name}_{frame_count:04d}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"{output_folder}/{video_name}_{frame_count:04d}.jpg")
            frame_count += 1

        # Đóng video và giải phóng tài nguyên
        print(f"{output_folder}/{video_name}_{frame_count:04d}.jpg")
        cap.release()

    cv2.destroyAllWindows()

# Thư mục đầu vào chứa các video
input_folder_path = 'D:/Tai lieu hoc/Nhap/IR/backend/dataset/input'

# Thư mục đầu ra cho các frame
output_folder_path = 'D:/Tai lieu hoc/Nhap/IR/backend/dataset/output'

# Gọi hàm để chuyển đổi video thành frame với target_fps = 20
video_to_frames(input_folder_path, output_folder_path, target_fps=1)