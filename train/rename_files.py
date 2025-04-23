import os


def rename_folders_and_files(dataset_path, old_names, new_names):
    # Đổi tên thư mục
    for old_name, new_name in zip(old_names, new_names):
        old_path = os.path.join(dataset_path, old_name)
        new_path = os.path.join(dataset_path, new_name)
        if os.path.exists(old_path):
            try:
                os.rename(old_path, new_path)
                print(f"Renamed folder: {old_path} -> {new_path}")
            except Exception as e:
                print(f"Error renaming folder {old_path}: {e}")
        else:
            print(f"Folder not found: {old_path}")

    # Đổi tên file trong mỗi thư mục
    for new_name in new_names:
        student_path = os.path.join(dataset_path, new_name)
        if not os.path.isdir(student_path):
            print(f"Directory not found: {student_path}")
            continue

        for idx, file in enumerate(os.listdir(student_path)):
            if file.endswith((".jpg", ".png")):
                old_path = os.path.join(student_path, file)
                # Tạo tên mới: <new_name>_<index>.jpg
                new_file = f"{new_name}_{idx + 1}.jpg"
                new_path = os.path.join(student_path, new_file)

                # Đổi tên file
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed file: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming file {old_path}: {e}")


# Cấu hình
dataset_path = "../team_data"
old_names = ["Trường", "Đan", "Trọng", "Dũng"]
new_names = ["Truong", "Dan", "Trong", "Dung"]

# Chạy đổi tên
rename_folders_and_files(dataset_path, old_names, new_names)