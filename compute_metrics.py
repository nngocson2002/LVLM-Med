import json
import re
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

CLASS_NAMES = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Clavicle fracture", "Consolidation", "Edema", "Emphysema", "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung cavity", "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Rib fracture", "Other lesion", "COPD", "Lung tumor", "Pneumonia", "Tuberculosis", "Other disease", "No finding", "Finding"]

def extract_labels(text):
    labels = [0] * len(CLASS_NAMES)
    text = re.sub(r"<.*?>", "", text.lower())
    for i, label in enumerate(CLASS_NAMES[:-1]):
        if label.lower() in text:
            labels[i] = 1
    if not labels[-2]:
        labels[-1] = 1
    return labels

def visualize_classification_report(gt_path, pred_path):
    """
        gt_path: path to csv file (e.g image_label_text.csv)
        pred_path: path to json file
    """
    gt = pd.read_csv(gt_path)
    gt["Finding"] = 1 - gt["No finding"]
    gt_vec = gt.iloc[:, 1:]

    data_pred = json.load(open(pred_path, 'r'))
    pred = {data["image_id"]: extract_labels(data["predict"]) for data in data_pred}
    pred_vec = np.vstack([pred[image_id] for image_id in gt["image_id"]])

    print(classification_report(gt_vec, pred_vec, target_names=CLASS_NAMES))