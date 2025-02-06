import pymongo
import numpy as np
from gridfs import GridFS
from io import BytesIO
from typing import List, Dict, Union

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



def load_images_from_mongodb(query: Dict = None, modality: str = None, body_part: str = None, is_validate : bool = True) -> List[Dict[str, Union[str, np.ndarray]]]:
    
    query = query or {} #Fetch all the query if query  None
    results = []
    collection_name = f"{"validation" if is_validate else "training"}_{modality}_{body_part}_images"
    fs = GridFS(db, collection=collection_name)
    
    try:
        cursor = db[collection_name].find(query)

        for doc in cursor:
            file_id = doc["file_id"]
            file_data = fs.get(file_id).read()
            image_array = np.load(BytesIO(file_data))

            image_info = {
                "filename": doc["file_name"],
                "image_array": image_array,
                "metadata": {key: value for key, value in doc.items() if key not in ["_id", "file_id"]} #Return metadata except id of the file in mongodb (id)
            }
            results.append(image_info)

        print("f{len(results) Image loaded from MongoDB}")
        return results

    except Exception as exception:
        print("Error while loading images: {exception}")
        return