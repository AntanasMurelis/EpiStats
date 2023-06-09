import numpy as np
from skimage import io
import os
import json
from types import SimpleNamespace
from typing import Union

#------------------------------------------------------------------------------------------------------------
def load_labeled_img(
        labeled_img: Union[str, np.ndarray]
    ) -> np.ndarray:
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
def create_output_directory(
        output_folder: str, 
        input_img_path: str,
        smoothing_iterations: int, 
        erosion_iterations: int, 
        dilation_iterations: int
    ) ->  str:
    """
    Create a directory to store the output data.

    Parameters:
        output_folder: (str) 
            The root folder for storing the outputs.
        input_img_path: (str)
            The path to the image to analyze, used to produce an
            identifier for the directory.
        smoothing_iterations: (int)
            The number of smoothing iterations in preprocessing.
        erosion_iterations: (int)
            The number of erosion iterations in preprocessing.
        dilation_iterations: (int)
            The number of dilation iterations in preprocessing.

    Returns:
        output_directory: (str) 
            The full path of the created output directory.
    """
    fname = os.path.basename(input_img_path).replace('.tif', '')
    output_directory = f"{output_folder}_{fname}_s_{smoothing_iterations}_e_{erosion_iterations}_d_{dilation_iterations}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_directory
#------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    """
    When printing a warning suppress everything except for the message and the category of the warning.
    """
    print(f"{category.__name__}: {message}")
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def read_config(path):
    """
    Read args from config.json file.

    Parameters:
    -----------
        path: (str)
            Path to the config.json file storing parameters for running the code.
        
    Returns:
    --------
        args: 
        A dictionary of parameters, whose attributes can be accessed by `args.my_attribute`.

    """
    with open(path) as f:
        args = json.load(f, object_hook = lambda d: SimpleNamespace(**d))

    return args

#------------------------------------------------------------------------------------------------------------
