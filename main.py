from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
# from google.cloud import storage
from fastapi.staticfiles import StaticFiles
import pill_identification
import math
import package_identification
import os
import numpy as np

print("Done loading OCR module.")
print("Starting server...")

app = FastAPI()

# Google Cloud Storage setup
# BUCKET_NAME = "pillow-images-uploads" 
# storage_client = storage.Client()
# bucket = storage_client.bucket(BUCKET_NAME)

app.mount("/package-images", StaticFiles(directory="MedsForAll_Images"), name="package-images")
app.mount("/pill-images", StaticFiles(directory="Pill Images Samples"), name="pill-images")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Define GCS file path
    # blob = bucket.blob(f"uploads/{file.filename}")
    
    # Upload file using upload_from_filename to preserve metadata
    temp_path = f"uploads/{file.filename}"  # Temporary local storage
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # blob.upload_from_filename(temp_path, content_type=file.content_type)
    
    # # Public URL of the uploaded image
    # file_url = f"https://storage.googleapis.com/{BUCKET_NAME}/uploads/{file.filename}"

    # downloaded_path = f"/tmp/downloaded_{file.filename}"
    # blob.download_to_filename(downloaded_path)

    # Get medication info
    med_info = package_identification.get_info(temp_path)
    #med_info = json.dumps(med_info, default=str)
    #return {"file_url": file_url, "ocr_text": ocr_text, "med_info": med_info}
    def clean_med_info(data):
        if isinstance(data, dict):
            return {k: clean_med_info(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_med_info(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            val = float(data)
            return None if math.isnan(val) or math.isinf(val) else val
        elif isinstance(data, float):
            return None if math.isnan(data) or math.isinf(data) else data
        elif isinstance(data, np.ndarray):
            return [clean_med_info(x) for x in data.tolist()]
        return data

    cleaned = clean_med_info(med_info)
    os.remove(temp_path)

    return JSONResponse(content=cleaned)

# def convert_numpy(obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return str(obj)

@app.post("/upload-pill-image/")
async def upload_image(file: UploadFile = File(...)):
    # Define GCS file path
    # blob = bucket.blob(f"uploads/{file.filename}")
    
    # Upload file using upload_from_filename to preserve metadata
    temp_path = f"uploads/{file.filename}"  # Temporary local storage
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # blob.upload_from_filename(temp_path, content_type=file.content_type)
    
    # Public URL of the uploaded image
    # file_url = f"https://storage.googleapis.com/{BUCKET_NAME}/uploads/{file.filename}"

    # Download file back before OCR (avoids URL issues)
    # downloaded_path = f"/tmp/downloaded_{file.filename}"
    # blob.download_to_filename(downloaded_path)

    med_info = pill_identification.get_info(temp_path)

    def clean_med_info(data):
        if isinstance(data, dict):
            return {k: clean_med_info(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_med_info(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            val = float(data)
            return None if math.isnan(val) or math.isinf(val) else val
        elif isinstance(data, float):
            return None if math.isnan(data) or math.isinf(data) else data
        elif isinstance(data, np.ndarray):
            return [clean_med_info(x) for x in data.tolist()]
        return data

    cleaned = clean_med_info(med_info)
    os.remove(temp_path)
    
    return JSONResponse(content=cleaned)





