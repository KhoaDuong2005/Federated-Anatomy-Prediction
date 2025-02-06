import pymongo
import numpy as np
from gridfs import GridFS
from io import BytesIO
from typing import List, Dict, Union
from data_loader import show_image

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["medical_db"]


def save_images_to_mongodb(
    images: List[np.ndarray], 
    file_names: List[str], 
    metadata: List[Dict] = None, 
    modality: str = None, 
    body_part: str = None, 
    is_validate: bool = True
):
    file_ids = []

    try:
        collection_name = "validation" if is_validate else "training"
        gridfs_collection_name = f"{collection_name}_{modality}_{body_part}_images"
        metadata_collection_name = f"{collection_name}_{modality}_{body_part}_metadata"

        fs = GridFS(db, collection=gridfs_collection_name)
        metadata_collection = db[metadata_collection_name]

        for i, image in enumerate(images):
            image_bytes = BytesIO()
            np.save(image_bytes, image)

            image_bytes.seek(0)

            file_id = fs.put(image_bytes.read(), file_name=file_names[i])
            file_ids.append(file_id)

            metadata_entry = {
                "file_name": file_names[i],
                "file_id": file_id,
                "dataset_type": collection_name,
                "modality": modality,
                "body_part": body_part
            }

            if metadata:
                metadata_entry.update(metadata[i])
            metadata_collection.insert_one(metadata_entry)
                
        print(f"{len(file_ids)} Images have been saved to MongoDB under {gridfs_collection_name}")
        return file_ids

    except Exception as exception:
        print(f"Error saving image: {exception}")
        return []


def load_images_from_mongodb(query: Dict = None, modality: str = None, body_part: str = None, is_validate: bool = True) -> List[Dict[str, Union[str, np.ndarray]]]:
    
    query = query or {}  # Fetch all if query is None
    results = []
    collection_name = f"{'validation' if is_validate else 'training'}_{modality}_{body_part}_images"
    
    fs = GridFS(db, collection=collection_name)

    try:
        cursor = db[f"{collection_name}.files"].find(query)

        for doc in cursor:
            file_id = doc["_id"]
            file_data = fs.get(file_id).read()  
            image_array = np.load(BytesIO(file_data))

            image_info = {
                "filename": doc["file_name"],
                "image_array": image_array,
            }
            results.append(image_info)

        print(f"{len(results)} Image(s) loaded from MongoDB")
        return results

    except Exception as exception:
        print(f"Error while loading images: {exception}")
        return []
        

def get_image_info(query, modality, body_part, is_validate):
    images_info = load_images_from_mongodb(query, modality, body_part, is_validate)
    return images_info



# metadata_collection_name = f"{"validation" if is_validate else "training"}_{modality}_{body_part}_metadata"
# metadata_entry = db[metadata_collection_name].find_one({"file_id": file_id})
# print(f"Metadata for file_id {file_id}: {metadata_entry}")