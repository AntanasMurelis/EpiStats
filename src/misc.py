import numpy as np
from skimage import io
import os
import pyclesperanto_prototype as cle
from typing import Union

#------------------------------------------------------------------------------------------------------------
def load_labeled_img(labeled_img: Union[str, np.ndarray]):
    """
    Load a labeled image.

    Parameters:
        labeled_img: If string, the path of the labeled image file. If numpy array, the actual labeled image.

    Returns:
        labeled_img: The loaded labeled image.
    """
    if isinstance(labeled_img, str) and os.path.isfile(labeled_img):
        labels = io.imread(labeled_img)
        labeled_img = np.einsum('kij->ijk', labels)
    return labeled_img
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def create_output_directory(output_folder: str, smoothing_iterations: int, erosion_iterations: int, dilation_iterations: int):
    """
    Create a directory to store the output data.

    Parameters:
        output_folder: The root folder for storing the outputs.
        smoothing_iterations: The number of smoothing iterations in preprocessing.
        erosion_iterations: The number of erosion iterations in preprocessing.
        dilation_iterations: The number of dilation iterations in preprocessing.

    Returns:
        output_directory: The full path of the created output directory.
    """
    output_directory = f"{output_folder}_s_{smoothing_iterations}_e_{erosion_iterations}_d_{dilation_iterations}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return output_directory
#------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    '''
    When printing a warning suppress everything except for the message and the category of the warning.
    '''
    print(f"{category.__name__}: {message}")