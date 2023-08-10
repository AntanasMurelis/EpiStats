import numpy as np
from tifffile import imwrite



"""
This scripts can be used to generate the 3D image of a sphere. The image can be used
as a ground truth to test the scripts that collect the statistics on the segmentation image.
"""


#------------------------------------------------------------------------------------------------------------
def generate_image_of_sphere(
    sphere_radius, 
    voxel_size
):
    """
    Generate a 3D image of a sphere. The image as a label of 1 for the sphere and 0 for the background.

    Parameters:
    -----------
        sphere_radius (float):
            The radius of the sphere in um

        voxel_size (float):
            The size of the voxel in um
    """

    # Calculate the size of the image in voxels
    image_size_in_voxels = int(2 * sphere_radius / voxel_size) + 1

    # Create the image
    image = np.zeros((image_size_in_voxels, image_size_in_voxels, image_size_in_voxels), dtype=np.uint8)

    # Calculate the center of the image
    center = int(image_size_in_voxels / 2)

    # Calculate the radius of the sphere in voxels
    sphere_radius_in_voxels = int(sphere_radius / voxel_size)

    # Add the sphere to the image
    for x in range(image_size_in_voxels):
        for y in range(image_size_in_voxels):
            for z in range(image_size_in_voxels):
                if (x - center)**2 + (y - center)**2 + (z - center)**2 <= sphere_radius_in_voxels**2:
                    image[x, y, z] = 1

    return image


#------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Generate the image of a sphere
    image = generate_image_of_sphere(sphere_radius=100, voxel_size=1)

    # Save the image
    imwrite("sphere.tif", image)
        