import numpy as np 
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2


def load_medical_image(file_path, size):
    try:
        image = sitk.ReadImage(file_path)

        image_array = sitk.GetArrayFromImage(image)
        
        # if (image_array.ndim == 3 and image_array[2] != 3):
        #     middle = image_array.shape[0] // 2
        #     image_array = image_array[middle, :, :]
        
        image_array = np.array(image_array)
        image_array = cv2.resize(image_array, size, interpolation=cv2.INTER_AREA)

        if image_array.ndim == 3:
            image_array = image_array[:, :, 0]

        image_array = np.stack([image_array] * 3, axis=0)

        return image_array
    except:
        raise ValueError("File not supported")

def show_image(image_array):
    plt.imshow(image_array[0,:,:], cmap = "gray")
    plt.show()

