import os
import shutil
import random

def replicate_images(source_dir, temp_dir, times=3):
    os.makedirs(temp_dir, exist_ok=True)
    class_names = os.listdir(source_dir)

    for class_name in class_names:
        src_class_dir = os.path.join(source_dir, class_name)
        dst_class_dir = os.path.join(temp_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        image_files = [f for f in os.listdir(src_class_dir) if f.lower().endswith('.jpg')]

        for img_file in image_files:
            src_img_path = os.path.join(src_class_dir, img_file)
            # Salin asli
            shutil.copy2(src_img_path, os.path.join(dst_class_dir, img_file))

            # Salin tambahan
            for i in range(1, times):
                new_filename = f"copy{i}_{img_file}"
                dst_img_path = os.path.join(dst_class_dir, new_filename)
                shutil.copy2(src_img_path, dst_img_path)

        print(f"[{class_name}] {len(image_files)} → {len(image_files) * times} duplikasi selesai")

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Rasio tidak valid"
    random.seed(seed)
    class_names = os.listdir(source_dir)

    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    for class_name in class_names:
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
        random.shuffle(images)

        total = len(images)
        train_count = int(train_ratio * total)
        val_count = int(val_ratio * total)
        test_count = total - train_count - val_count

        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count+val_count]
        test_imgs = images[train_count+val_count:]

        for img in train_imgs:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(output_dir, 'train', class_name, img))
        for img in val_imgs:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(output_dir, 'val', class_name, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(output_dir, 'test', class_name, img))

        print(f"[{class_name}] Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# Eksekusi
if __name__ == "__main__":
    replicate_images("archive", "archive_replicated", times=2)
    split_dataset("archive_replicated", "archive_replicated_result", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
