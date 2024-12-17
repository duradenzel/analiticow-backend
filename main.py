import logging
from fastapi import FastAPI, File, Query, UploadFile
import cv2
import numpy as np
from utils import load_yolo_model, run_inference, read_text, crop_image, load_dataset, find_closest_match
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_yolo_model("best.pt")
dataset = load_dataset("Stallijst Baarlesebaan 30-8-24.csv", column_name="Levensnummer")
video_dataset = load_dataset("video_dataset.csv", column_name="Levensnummer")  # New dataset for video frames

@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1].lower()
    logger.info(f"Received file: {file.filename}, file extension: {file_extension}")
    
    contents = await file.read()
    results = []

    if file_extension in ["jpg", "jpeg", "png", "bmp"]:
        logger.info("Processing image file")
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        results = process_single_image(image, dataset)  

    elif file_extension in ["mp4", "avi", "mov", "mkv"]:
        logger.info("Processing video file")
        temp_file = "temp_video." + file_extension
        with open(temp_file, "wb") as f:
            f.write(contents)
        results = process_video(temp_file, video_dataset)  

    else:
        logger.error(f"Unsupported file format: {file_extension}")
        return {"error": "Unsupported file format"}

    logger.info(f"Processing completed with {len(results)} results")
    return {"results": results}


def process_single_image(image, csv):
    logger.debug("Running inference on the image")
    results = run_inference(model, image)
    ocr_results = []

    for result in results:
        for box in result.boxes:
            cropped_image = crop_image(image, box)
            ocr_text = read_text(cropped_image)

            logger.debug(f"OCR text: {ocr_text}")

            closest_match, distance, closest_row = find_closest_match(ocr_text, csv)

            ocr_results.append({
                "text": ocr_text,
                "bbox": box.xyxy.tolist(),
                "closest_match": closest_match,
                "distance": distance,
                "record": closest_row
            })

    logger.debug(f"Found {len(ocr_results)} OCR results in the image")
    return ocr_results


def process_video(video_path, csv):
    logger.debug(f"Processing video: {video_path}")
    
    output_folder = "processed_frames"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  
    os.makedirs(output_folder) 
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file")
        return []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate)
    logger.info(f"Frame rate: {frame_rate} FPS, processing every {frame_interval}th frame")

    ocr_results = []
    frame_count = 0
    saved_frame_count = 0  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            logger.debug(f"Processing frame {frame_count}")
            
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            logger.info(f"Saved frame {frame_count} to {frame_filename}")
            saved_frame_count += 1
            
            results = process_single_image(frame, csv)
            ocr_results.extend(results)

        frame_count += 1

    cap.release()
    logger.info(f"Processed {frame_count} total frames, saved {saved_frame_count} frames, "
                f"and found {len(ocr_results)} results")
    return ocr_results

@app.post("/search/")
def search_record(data: dict):
    query = data.get("query", "")
    if not query:
        return {"error": "Query is required"}
    result = []
    closest_entry, distance, closest_row = find_closest_match(query, dataset)
    result.append({
        "query": query,
        "closest_match": closest_entry,
        "distance": distance,
        "record": closest_row
    })
    print(result)
    
    return {"result": result}