from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from utils import load_yolo_model, run_inference, read_text, crop_image, load_dataset, find_closest_match
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Ear Tag Recognition API"}

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    results = run_inference(model, image)
    ocr_results = []

    for result in results:
        for box in result.boxes:
            cropped_image = crop_image(image, box)

            ocr_text = read_text(cropped_image)

            closest_match, distance, closest_row = find_closest_match(ocr_text, dataset)

            ocr_results.append({
                "text": ocr_text,
                "bbox": box.xyxy.tolist(),  
                "closest_match": closest_match, 
                "distance": distance,
                "record": closest_row

            })

    return {"ocr_results": ocr_results}
