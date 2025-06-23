import os
import json
import numpy as np

SRC_DIR = "1_data/New_sample/LabelData/REAL/WORD/01_real_word_keypoint"
DST_DIR = "1_data/processed/sequence_data"

os.makedirs(DST_DIR, exist_ok=True)

print(f"SRC_DIR: {os.path.abspath(SRC_DIR)}")
print(f"DST_DIR: {os.path.abspath(DST_DIR)}")

for label in os.listdir(SRC_DIR):
    label_dir = os.path.join(SRC_DIR, label)
    if not os.path.isdir(label_dir):
        print(f"Skip {label_dir} (not a dir)")
        continue
    save_label_dir = os.path.join(DST_DIR, label)
    os.makedirs(save_label_dir, exist_ok=True)
    print(f"Processing label: {label}")

    keypoints_seq = []
    files = sorted(os.listdir(label_dir))
    for file in files:
        if not file.endswith(".json"):
            continue
        json_path = os.path.join(label_dir, file)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 예시: hand_left_keypoints_2d 또는 pose_keypoints_2d 등에서 21*3=63개 추출
                # 아래는 hand_left_keypoints_2d 사용 예시
                kps = data["people"]["hand_left_keypoints_2d"][:63]
                keypoints_seq.append(kps)
        except Exception as e:
            print(f"    Error reading {json_path}: {e}")
    if keypoints_seq:
        keypoints_seq = np.array(keypoints_seq)  # (프레임수, 63)
        save_path = os.path.join(save_label_dir, label + ".npy")
        np.save(save_path, keypoints_seq)
        print(f"    Saved: {save_path}, shape={keypoints_seq.shape}")
    else:
        print(f"    No keypoints found in {label_dir}")
