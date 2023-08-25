# EpiStats: collection of morphological cell statistics from segmented 3D images of epithelial tissues

This repository contains tools for computing and plotting morphological statistics of cells in 3D images of epithelial tissues. It also provides methods to transform the segmented cells into mesh files and to fix these meshes in case they present any issue.

<table border="0">
  <tr>
    <td><img src="https://github.com/AntanasMurelis/EpiStats/blob/main/images/segmentation.png" alt="Segmented image" width="600"></td>
    <td><img src="https://github.com/AntanasMurelis/EpiStats/blob/main/images/meshes_group.png" alt="Meshes" width="600"></td>
  </tr>
</table>

## INSTALLATION:
Install venv if not already done
```bash
pip3 install virtualenv
```

Create a virtual environment
```bash
python3 -m venv EpiStatsVenv
```

Activate the virtual env
```bash
source ./EpiStatsVenv/bin/activate
```

Install the required packages
```bash
pip3 install -r requirements.txt
```


## USAGE:
- Scripts for morphological statistics computation, data post-processing, and plotting is in
  ```
  src/statistics_collection/
  ```
- Scripts for mesh refinement is in
  ```
  src/mesh_fixing
  ```
 
