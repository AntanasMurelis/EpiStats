# EpiStats: collection of morphological cell statistics from segmented 3D images of epithelial tissues

This repository contains tools for computing and plotting morphological statistics of cells in 3D images of epithelial tissues. It also provides methods to transform the segmented cells into mesh files and to fix these meshes in case they present any issue.

## Installation
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

