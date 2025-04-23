import imgaug.augmenters as iaa
import cv2
import os

seq = iaa.Sequential([
    iaa.Affine(rotate=(-10, 10)),
    iaa.MultiplyBrightness((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0, 0.5))
])

for subdir in os.listdir("../team_data"):
    subdir_path = os.path.join("dataset", subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.endswith((".jpg", ".png")):
                image = cv2.imread(os.path.join(subdir_path, file))
                aug_images = seq(images=[image] * 5)  # Tạo 5 ảnh biến thể
                for i, aug_img in enumerate(aug_images):
                    cv2.imwrite(os.path.join(subdir_path, f"aug_{i}_{file}"), aug_img)