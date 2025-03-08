from fastapi import FastAPI, File, UploadFile
import shutil
import os
import json

import medication_matching
import ocr_module

print("Done loading OCR module.")
print("Starting server...")

app = FastAPI()

#uvicorn main:app --host 0.0.0.0 --port 8000
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create directory if it doesn't exist
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    image_path = f'uploads/{file.filename}'
    ocr_text = ocr_module.extract_image_text(image_path=image_path)

    med_info = medication_matching.get_info(ocr_text)
    med_info = json.dumps(med_info, default=str)
    #return {"filename": file.filename, "content_type": file.content_type}
    return med_info
