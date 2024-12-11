from ultralytics import YOLO
import easyocr
import pandas as pd
from Levenshtein import distance as levenshtein_distance
import cv2


def load_yolo_model(model_path):
    return YOLO(model_path)

def run_inference(model, image):
    return model.predict(source=image, save=False, verbose=False)

reader = easyocr.Reader(['en'])

def read_text(image):
    try:
        results = reader.readtext(image, detail=0)
        return " ".join(results)
    except Exception as e:
        print(f"Error reading text: {e}")
        return "N/A"

def crop_image(image, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return image[y1:y2, x1:x2]


def load_dataset(csv_path, column_name):
    data = pd.read_csv(csv_path)
    return data[column_name].astype(str).tolist()

def find_closest_match(ocr_text, dataset_entries):
    closest_entry = None
    min_distance = float('inf')
    for entry in dataset_entries:
        distance = levenshtein_distance(ocr_text, entry)
        if distance < min_distance:
            min_distance = distance
            closest_entry = entry
    return closest_entry, min_distance
