from flask import Flask, render_template, request, jsonify
import json
import sys
import os

import trimesh
import numpy as np
from shapely.geometry import Polygon


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

def meter_to_mm(meters):
    return round(meters * 1000.0, 2)

def mm_to_meter(mm):
    return round(mm / 1000.0, 2)


def convert_measure(measure_in_meters, target_unit='mm'):
    """
    Convert a measure from meters to the specified target unit.

    Parameters:
    - measure_in_meters (float): The measure in meters.
    - target_unit (str): The target unit for conversion. Default is 'cm'.
                        Possible values: 'in', 'cm', 'mm', or any other unit.

    Returns:
    - float: The converted measure in the target unit.
    """
    # Define conversion factors
    conversion_factors = {
        'in': 39.3701,   # 1 meter = 39.3701 inches
        'cm': 100,       # 1 meter = 100 centimeters
        'mm': 1000,      # 1 meter = 1000 millimeters
        # Add more units as needed
    }

    # Check if the target unit is valid
    if target_unit not in conversion_factors:
        return 0.0

    # Perform the conversion
    converted_measure = measure_in_meters * conversion_factors[target_unit]
    
    return round(converted_measure,2)

def slice_3d_model(plane_origin,plane_normal,color,mesh):
  # Perform the slicing using the slice_plane method
  slice_mesh = mesh.slice_plane(plane_origin, plane_normal)
  # Set the color of the sliced mesh
  color = color
  slice_mesh = trimesh.Trimesh(vertices=slice_mesh.vertices,
                             faces=slice_mesh.faces,
                             face_colors=color)
  return slice_mesh


def get_circular_length(specific_height,mesh,plane_origin,plane_normal):
   slice_mesh = mesh.section(plane_origin=plane_origin,plane_normal=plane_normal)
   s_mesh = trimesh.Trimesh(vertices=slice_mesh.vertices)
   hull_vertices = s_mesh.convex_hull.vertices
   #Get convex hull as a new Trimesh object
   hull_mesh = s_mesh.convex_hull
   # Create a closed path using the vertices
   points = hull_mesh.vertices[:, [0, 2]]
   # Compute the convex hull
   hull = ConvexHull(points)
   # Get the perimeter (length) of the convex hull
   perimeter_length = hull.area

   return perimeter_length







app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file part'})

    # file = request.files['file']

    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'})

    # if file:
    #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #     file.save(file_path)
    
    # # Load your 3D model
    # mesh = trimesh.load_mesh(file_path)
    # Access JSON data from the POST request
    json_data = request.get_json()
    print(json_data)
    metric = 'in'

    crotch_point = 0.7418
    knee_point = 0.51800
    ankle_point = 0.08423 

    crotch_plane_origin, crotch_plane_normal =[0, crotch_point, 0], [0, 1, 0]
    knee_plane_origin, knee_plane_normal =[0, knee_point, 0], [0, 1, 0]
    ankle_plane_origin, ankle_plane_normal =[0, ankle_point, 0], [0, 1, 0]


    upper_thigh =get_circular_length(crotch_point,mesh,crotch_plane_origin,crotch_plane_normal)
    knee =get_circular_length(knee_point,mesh,knee_plane_origin,knee_plane_normal)
    ankle =get_circular_length(ankle_point,mesh,ankle_plane_origin,ankle_plane_normal)
   
   # Prepare the measurement data in JSON format
    measurement_data = {
        'upper_thigh':convert_measure(upper_thigh, metric),
        'knee': convert_measure(knee, metric),
        'ankle': convert_measure(ankle, metric)
    }

    # Return the measurements as JSON response
    return jsonify(measurement_data)


if __name__ == '__main__':
    app.run(debug=True)