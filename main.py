from fastapi import FastAPI, File, UploadFile
from google.cloud import storage
import json
import medication_matching
import ocr_module
import os

print("Done loading OCR module.")
print("Starting server...")

app = FastAPI()

# Google Cloud Storage setup
BUCKET_NAME = "pillow-images-uploads"  # Replace with your actual GCS bucket name
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Define GCS file path
    blob = bucket.blob(f"uploads/{file.filename}")
    
    # Upload file using upload_from_filename to preserve metadata
    temp_path = f"/tmp/{file.filename}"  # Temporary local storage
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    blob.upload_from_filename(temp_path, content_type=file.content_type)
    
    # Public URL of the uploaded image
    file_url = f"https://storage.googleapis.com/{BUCKET_NAME}/uploads/{file.filename}"

    # Download file back before OCR (avoids URL issues)
    downloaded_path = f"/tmp/downloaded_{file.filename}"
    blob.download_to_filename(downloaded_path)

    # Perform OCR on the downloaded image
    ocr_text = ocr_module.extract_image_text(image_path=downloaded_path)

    # Get medication info
    med_info = medication_matching.get_info(ocr_text)
    med_info = json.dumps(med_info, default=str)

    #return {"file_url": file_url, "ocr_text": ocr_text, "med_info": med_info}
    return med_info


# from fastapi import FastAPI, File, UploadFile
# import shutil
# import os
# import json

# import medication_matching
# import ocr_module

# print("Done loading OCR module.")
# print("Starting server...")

# app = FastAPI()

# #uvicorn main:app --host 0.0.0.0 --port 8000
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create directory if it doesn't exist
# @app.post("/upload-image/")
# async def upload_image(file: UploadFile = File(...)):
#     file_path = f"{UPLOAD_DIR}/{file.filename}"
    
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     image_path = f'uploads/{file.filename}'
#     ocr_text = ocr_module.extract_image_text(image_path=image_path)
#     print(ocr_text)
#     med_info = medication_matching.get_info(ocr_text)
#     med_info = json.dumps(med_info, default=str)
#     #return {"filename": file.filename, "content_type": file.content_type}
#     return med_info