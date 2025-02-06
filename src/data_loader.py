import numpy as np 
import SimpleITK as sitk
import matplotlib.pyplot as plt


def load_medical_image(file_path):
    try:
        image = sitk.ReadImage(file_path)
        
        image_array = sitk.GetArrayFromImage(image)
        
        # if image_array.ndim == 3:
        #     middle = image_array.shape[0] // 2
        #     image_array = image_array[middle, :, :]
        
        image_array = np.array(image_array)

        return image_array
    except:
        raise ValueError("File not supported")

def show_image(image_array):
    plt.imshow(image_array, cmap = "gray")
    plt.show()

