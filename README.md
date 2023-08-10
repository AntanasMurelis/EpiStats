# EpiStats: collection of morphological cell statistics from segmented 3D images of epithelial tissues

This repository contains tools for computing and plotting morphological statistics of cells in 3D images of epithelial tissues. It also provides methods to transform the segmented cells into mesh files and to fix these meshes in case they present any issue.

<table>
  <tr>
    <td><img src="./results/screening_dashboard_type_1.jpg" alt="Type 1 dashboard"></td>
    <td><img src="./results/screening_dashboard_type_2.jpg" alt="Type 2 dashboard"></td>
    <td><img src="./results/screening_dashboard_type_2.jpg" alt="Type 2 dashboard"></td>
  </tr>
</table>

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

