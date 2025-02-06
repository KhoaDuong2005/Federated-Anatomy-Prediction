import pymongo
import numpy as np
from gridfs import GridFS
from io import BytesIO
from typing import List, Dict

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["medical_db"]
fs = GridFS(db)
collection = db["images"]

def save_images_to_mongodb(images: List[np.ndarray], file_names: List[str], metadata: List[Dict] = None):
    file_ids = []
    try:
        for i, image in enumerate(images):
            image_bytes = BytesIO()
            np.save(image_bytes, image)

            file_id = fs.put(image_bytes.read(), file_name = file_names[i])
            file_ids.append(file_id)

            data = {"file_name": file_names[i], "file_id": file_id}
            
            #check if metadata exist
            if metadata:
                data.update(metadata[i])
                
        print("{len(file_ids) Images has been saved to MongoDB}")
        return file_ids

    except Exception as exception:
        print(f"Error saving image: {e}")
        return []

def load_images_from_mongodb(query: Dict = None):
    pass