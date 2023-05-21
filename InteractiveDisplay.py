import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QComboBox, QWidget
import pandas as pd
import sys
import os
import numpy as np
# class Window(QMainWindow):
#     def __init__(self, mesh_dir, statistics_csv):
#         super().__init__()

#         # Load the STL files
#         self.mesh = pv.MultiBlock()
#         for filename in os.listdir(mesh_dir):
#             if filename.endswith('.stl'):
#                 self.mesh.append(pv.read(os.path.join(mesh_dir, filename)))

#         # Load the cell statistics
#         self.statistics_df = pd.read_csv(statistics_csv)

#         # Create a PyVista plotter
#         self.plotter = QtInteractor(self)
        
#         # Create a dropdown menu with the statistic columns
#         self.dropdown = QComboBox(self)
#         for column in self.statistics_df.columns:
#             self.dropdown.addItem(column)

#         # Connect the dropdown menu to a function
#         self.dropdown.activated[str].connect(self.on_dropdown_activated)

#         # Create a central widget
#         central_widget = QWidget()

#         # Create a layout
#         layout = QVBoxLayout()

#         # Add the dropdown and plotter to the layout
#         layout.addWidget(self.dropdown)
#         layout.addWidget(self.plotter.interactor)

#         # Set the layout of the central widget
#         central_widget.setLayout(layout)

#         # Set the central widget of the QMainWindow
#         self.setCentralWidget(central_widget)

#         # Show the window
#         self.show()
        
#          # Add meshes to the plotter initially
#         for mesh in self.mesh:
#             self.plotter.add_mesh(mesh)
#         self.plotter.reset_camera()


    # def on_dropdown_activated(self, text):
    #     # Get the statistic values for the cells
    #     cell_statistic_values = self.statistics_df[text].values.astype(float)

    #     # Normalize the statistic values to the range [0, 1]
    #     normalized_statistic_values = (cell_statistic_values - cell_statistic_values.min()) / (cell_statistic_values.max() - cell_statistic_values.min())

    #     # Update the meshes with the new scalars
    #     for i, mesh in enumerate(self.mesh):
    #         mesh.point_data[text] = normalized_statistic_values[i]

    #     # Update the plotter with the new meshes
    #     self.plotter.clear()
    #     for mesh in self.mesh:
    #         self.plotter.add_mesh(mesh, scalars=text, show_scalar_bar=True)
    #     self.plotter.reset_camera()

import trimesh

# class Window(QMainWindow):
#     def __init__(self, mesh_dir, statistics_csv):
#         super().__init__()

#         # Load the STL files
#         self.mesh = pv.MultiBlock()
#         self.statistics_df = pd.read_csv(statistics_csv)

#         # Get cell IDs and volumes from CSV file
#         csv_cell_ids = self.statistics_df['cell_id'].values
#         csv_cell_volumes = self.statistics_df['cell_volume'].values

#         for filename in os.listdir(mesh_dir):
#             if filename.endswith('.stl'):
#                 # Extract cell_id from filename assuming it's in the form 'cell_{cell_id-1}.stl'
#                 cell_id = int(filename.split('_')[1].split('.')[0]) + 1
                
#                 # Check if cell_id is in the CSV file and its volume is greater than 200
#                 if cell_id in csv_cell_ids and csv_cell_volumes[csv_cell_ids == cell_id][0] > 200:
#                     self.mesh.append(pv.read(os.path.join(mesh_dir, filename)))

#         # Create a PyVista plotter
#         self.plotter = QtInteractor(self)

#         # Create a dropdown menu with the statistic columns
#         self.dropdown = QComboBox(self)
#         for column in self.statistics_df.columns:
#             self.dropdown.addItem(column)

#         # Connect the dropdown menu to a function
#         self.dropdown.activated[str].connect(self.on_dropdown_activated)

#         # Create a central widget
#         central_widget = QWidget()

#         # Create a layout
#         layout = QVBoxLayout()

#         # Add the dropdown and plotter to the layout
#         layout.addWidget(self.dropdown)
#         layout.addWidget(self.plotter.interactor)

#         # Set the layout of the central widget
#         central_widget.setLayout(layout)

#         # Set the central widget of the QMainWindow
#         self.setCentralWidget(central_widget)

#         # Show the window
#         self.show()

#         # Add meshes to the plotter initially
#         for mesh in self.mesh:
#             self.plotter.add_mesh(mesh)
#         self.plotter.reset_camera()

    # def on_dropdown_activated(self, text):
    #     # Get the statistic values for the cells
    #     cell_statistic_values = self.statistics_df[text].values.astype(float)

    #     # Normalize the statistic values to the range [0, 1]
    #     normalized_statistic_values = (cell_statistic_values - cell_statistic_values.min()) / (cell_statistic_values.max() - cell_statistic_values.min())

    #     # Update the meshes with the new scalars
    #     for i, mesh in enumerate(self.mesh):
    #         mesh.point_data[text] = normalized_statistic_values[i]

    #     # Update the plotter with the new meshes
    #     self.plotter.clear()
    #     for mesh in self.mesh:
    #         self.plotter.add_mesh(mesh, scalars=text, show_scalar_bar=True)
    #     self.plotter.reset_camera()

import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QComboBox, QWidget
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import sys
import os

import numpy as np
import pandas as pd
import os
import pyvista as pv
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QSizePolicy, QComboBox, QWidget
from PyQt5.QtCore import Qt
from pyvistaqt import QtInteractor

class Window(QMainWindow):
    def __init__(self, mesh_dir, statistics_csv):
        super().__init__()

        # Load the STL files
        self.mesh = pv.MultiBlock()
        self.statistics_df = pd.read_csv(statistics_csv)

        # Filter out cells with volume < 200
        self.statistics_df = self.statistics_df[self.statistics_df['cell_volume'] >= 200]

        # Get cell IDs from CSV file
        csv_cell_ids = self.statistics_df['cell_id'].values

        for filename in os.listdir(mesh_dir):
            if filename.endswith('.stl'):
                # Extract cell_id from filename assuming it's in the form 'cell_{cell_id-1}.stl'
                cell_id = int(filename.split('_')[1].split('.')[0]) + 1
                
                # Check if cell_id is in the CSV file
                if cell_id in csv_cell_ids:
                    self.mesh.append(pv.read(os.path.join(mesh_dir, filename)))

        # Make sure self.mesh and self.statistics_df include the same cells
        self.mesh = self.mesh[:len(self.statistics_df)]
        
        # Create a PyVista plotter
        self.plotter = QtInteractor(self)

        # Create a dropdown menu with the statistic columns
        self.dropdown = QComboBox(self)
        for column in self.statistics_df.columns:
            self.dropdown.addItem(column)

        # Connect the dropdown menu to a function
        self.dropdown.activated[str].connect(self.on_dropdown_activated)

        # Create a central widget
        central_widget = QWidget()

        # Create a layout
        layout = QVBoxLayout()

        # Add the dropdown and plotter to the layout
        layout.addWidget(self.dropdown)
        layout.addWidget(self.plotter.interactor)

        # Create a figure and canvas for matplotlib plotting
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        # Set the layout of the central widget
        central_widget.setLayout(layout)

        # Set the central widget of the QMainWindow
        self.setCentralWidget(central_widget)

        # Show the window
        self.show()

        # Add meshes to the plotter initially
        for mesh in self.mesh:
            self.plotter.add_mesh(mesh)
        self.plotter.reset_camera()

    def on_dropdown_activated(self, text):
        # Get the statistic values for the cells
        cell_statistic_values = self.statistics_df[text].values.astype(float)

        # Normalize the statistic values to the range [0, 1]
        normalized_statistic_values = (cell_statistic_values - cell_statistic_values.min()) / (cell_statistic_values.max() - cell_statistic_values.min())

        # Update the meshes with the new scalars
        for i, mesh in enumerate(self.mesh):
            mesh.point_data[text] = normalized_statistic_values[i]

        # Update the plotter with the new meshes
        self.plotter.clear()
        for mesh in self.mesh:
            self.plotter.add_mesh(mesh, scalars=text, show_scalar_bar=True)
        self.plotter.reset_camera()
        
        # Update the matplotlib graph
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist(normalized_statistic_values, bins=50, color='blue', alpha=0.7)
        ax.set_title(f'Histogram of {text}')
        ax.set_xlabel('Normalized value')
        ax.set_ylabel('Frequency')
        self.canvas.draw()




if __name__ == "__main__":
    # Replace these with the appropriate paths
    vtk_file = '/Users/antanas/BC_Project/Control_Segmentation_final/BC_control_2_s_8_e_3_d_4/cell_meshes'
    statistics_csv = '/Users/antanas/BC_Project/Control_Segmentation_final/BC_control_2_s_8_e_3_d_4/filtered_cell_statistics.csv'
    statistic_column = 'cell_volume'

    app = QApplication(sys.argv)
    window = Window(vtk_file, statistics_csv)
    sys.exit(app.exec_())
