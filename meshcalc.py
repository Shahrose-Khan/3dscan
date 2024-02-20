from flask import Flask, render_template, request, jsonify
import json
import sys
import os

import trimesh
import numpy as np
from shapely.geometry import Polygon


import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        'm':1
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


def get_circular_length(specific_height,mesh):
   slice_mesh = mesh.section(plane_origin=[0,specific_height,0],plane_normal=[0,1,0])
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

   return perimeter_length, s_mesh



def plot_mesh(ax, mesh_vertices, perimeter_length, height,metric , name):
    # Define the color names
    colors = {
    'A': ['orange', 'darkorange', (1.0, 0.8, 0.6)],
    'B': ['orange', 'darkorange', (1.0, 0.8, 0.6)],
    'C': ['orange', 'darkorange', (1.0, 0.8, 0.6)],
    'D': ['orange', 'darkorange', (1.0, 0.8, 0.6)],
    'E': ['orange', 'darkorange', (1.0, 0.8, 0.6)],
    'F': ['orange', 'darkorange', (1.0, 0.8, 0.6)],
    'G': ['orange', 'darkorange', (1.0, 0.8, 0.6)],
    }
    # Plot the points
    ax.scatter(mesh_vertices[:, 0], mesh_vertices[:, 2], marker='o', color=colors[name][0])

    # Create a closed path using the vertices
    points = mesh_vertices[:, [0, 2]]

    # Compute the convex hull
    hull = ConvexHull(points)

    # Plot the convex hull edges
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1],  color=colors[name][1])

    # Fill the region within the convex hull
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], color=colors[name][2])

    # Remove the axis
    ax.axis('off')


    # Calculate the y-coordinate for the split line
    y_split = max(mesh_vertices[:, 2]) + 0.1 * (max(mesh_vertices[:, 2]) - min(mesh_vertices[:, 2]))

    if(name != 'A'):
        # Add a horizontal split line
        ax.axhline(y=y_split, color='black')

    # Add a heading at the bottom
    ax.text(0.5, -0.15, f'C{name} = {convert_measure(perimeter_length, metric)} {metric}   |  L{name} = {convert_measure(height, metric)} {metric}',
            horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

    


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Define your mesh plotting function
def plot_meshes(mesh_vertices, perimeter_lengths, heights,metric ):

    # Create a single plot with subplots
    fig, axs = plt.subplots(7, 1, figsize=(5, 30))

    i = 0
    # Loop over heights and plot each graph
    for name, perimeter_length in perimeter_lengths.items():
        plot_mesh(axs[i], mesh_vertices[name].vertices, perimeter_length, heights[name],metric, name)
        i = i+1



    # Adjust layout
    plt.tight_layout()

    save_path = 'static/mesh_plots.png'
    # Save the plot as PNG image
    plt.savefig(save_path)
    # Do not display the plot
    plt.close()
    return save_path


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

    # Access the array data from the request
    # data_array = json.loads(request.form.getlist('parameters')[0])
    # print("Received Data Array:", round(data_array['LG'],4) )
    
    # Load your 3D model
    mesh = trimesh.load_mesh(file_path)
    
    metric = 'mm'
    # Three main points on the leg

    # LG =round(data_array['LG'],4)        #crotch_point
    # LE =round(data_array['LE'],4)         #knee_point
    # LA =round(data_array['LA'],4)        #ankle_point

    heights = {
    'A': 0.159,
    'B': 0.249, 
    'C': 0.328, 
    'D': 0.418, 
    'E': 0.498, 
    'F': 0.628, 
    'G': 0.758
    }

    perimeter_lengths = {}
    mesh_vertices = {}


    i = 0
    # Loop over heights and plot each graph
    for name, height in heights.items():
        perimeter_lengths[name], mesh_vertices[name] = get_circular_length(height, mesh)
    image_path = plot_meshes(mesh_vertices,perimeter_lengths,heights, metric )



   # Prepare the measurement data in JSON format
    measurement_data = {
        'metric' : metric,
        # 'image_url': f"http://127.0.0.1:5000/{image_path}",
        'CG' : convert_measure(perimeter_lengths['G'], metric),
        'CF' : convert_measure(perimeter_lengths['F'], metric),
        'CE' : convert_measure(perimeter_lengths['E'], metric),
        'CD' : convert_measure(perimeter_lengths['D'], metric),
        'CC' : convert_measure(perimeter_lengths['C'], metric),
        'CB' : convert_measure(perimeter_lengths['B'], metric),
        'CA' : convert_measure(perimeter_lengths['A'], metric),
        'LG' : convert_measure(heights['G'], metric),
        'LF' : convert_measure(heights['F'], metric),
        'LE' : convert_measure(heights['E'], metric),
        'LD' : convert_measure(heights['D'], metric),
        'LC' : convert_measure(heights['C'], metric),
        'LB' : convert_measure(heights['B'], metric),
        'LA' : convert_measure(heights['A'], metric),
    }
  
    # Return the measurements as JSON response
    return jsonify(measurement_data)

@app.route('/test')
def test():
    return jsonify({'success': 'true'})

if __name__ == '__main__':
    app.run(debug=True)