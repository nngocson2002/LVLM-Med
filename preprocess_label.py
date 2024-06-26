import pandas as pd
import os
import glob
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import json

IMAGE_FOLDER = "path/to/image_folder"

def get_image_size(path):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    image_id = os.path.basename(path).split(".")[0]
    return image_id, width, height

def preprocess_label(path, train=True):
    df = pd.read_csv(path)
    paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))

    images_size = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_image_size, path) for path in paths]
        
        for future in tqdm(as_completed(futures), total=len(paths)):
            image_id, width, height = future.result()
            images_size[image_id] = [width, height]

    sizes = np.array([images_size[image_id] for image_id in df["image_id"]])
    sizes = np.tile(sizes, 2)
    if train:
        df.iloc[:,3:] = round(df.iloc[:,3:] / sizes * 100)
    else:
        df.iloc[:,2:] = round(df.iloc[:,2:] / sizes * 100)

    df["labels"] = df.apply(lambda x: x["class_name"] if x["class_name"] == "No finding" else f"<p>{x['class_name']}</p> {{<{int(x['x_min'])}><{int(x['y_min'])}><{int(x['x_max'])}><{int(x['y_max'])}>}}", axis=1)
    df_labels = df.groupby('image_id')['labels'].apply(lambda x: ','.join(set(x))).reset_index()
    results = [{"image_id": image_id, "grounded_diseases": label} for image_id, label in zip(df_labels['image_id'], df_labels['labels'])]
    return results

if __name__ == "__main__":
    path = "path/to/train.csv"
    results = preprocess_label(path)

    with open('grounded_diseases_train.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)