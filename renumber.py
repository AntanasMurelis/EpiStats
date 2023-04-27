from SegmentationStatisticsCollector import renumber_labels
import argparse
import numpy as np
import os
from skimage import io
from tqdm import tqdm

# Add the existing function `renumber_labels` here

def main():
    parser = argparse.ArgumentParser(description="Renumber labels in a labeled image.")
    parser.add_argument("input_image", type=str, help="Path to the input labeled image file (in .tif or .npy format)")
    parser.add_argument("output_image", type=str, help="Path to the output renumbered image file (in .tif format)")
    args = parser.parse_args()

    # Read the input image
    if args.input_image.endswith(".tif"):
        labeled_img = io.imread(args.input_image)
    elif args.input_image.endswith(".npy"):
        labeled_img = np.load(args.input_image)
    else:
        raise ValueError("Unsupported input image format. Only .tif and .npy are supported.")

    # Renumber the labels
    renumbered_labeled_img = renumber_labels(labeled_img)

    # Save the renumbered labeled image to the output file in .tif format
    io.imsave(args.output_image, renumbered_labeled_img.astype(np.uint16))

if __name__ == "__main__":
    main()
