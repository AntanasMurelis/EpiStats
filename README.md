This scripts can be used on a 3D segmentation image to collect the morphological features of the cells. 

## Installation
Install venv if not already done
```pip3 install virtualenv```

Create a virtual environment
```python3 -m venv collect_cell_stats```

Activate the virtual env
```source ./collect_cell_stats/bin/activate```

Install the required packages
```pip3 install -R requirements.txt```

## Note
- When tested against the cubic test geometry, the measurement of the area introduces a ~5% error from the  real area. 
- In the same conditions the measurement of the volume introduces a ~1.5% error from the real volume.
- The contact fraction introduces a ~2% error from the real value.