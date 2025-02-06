import pymongo
import numpy as np
from gridfs import GridFS
from io import BytesIO
from typing import List, Dict

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["medical_db"]
fs = GridFS(db)
fs_train = gridfs.GridFS(db, collection="training_images")
fs_val = gridfs.GridFS(db, collection="validation_images")


def save_images_to_mongodb(images: List[np.ndarray], file_names: List[str], metadata: List[Dict] = None, is_train: bool = True):
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

def load_images_from_mongodb(query: Dict = None) -> List(Dict[str, Union[str, np.ndarray]]):
    
    query = query or {} #Fetch all the query if query  None
    results = []
    cursor = collection.find(querry)

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
