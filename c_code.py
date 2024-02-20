from flask import Flask, render_template, request, jsonify
from flask import send_file
import pymysql

import json
import sys
import os
from datetime import datetime

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

connection = pymysql.connect(host='127.0.0.1',
                                user='root',
                                password='root',
                                db='3dscan_db')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scans/<int:scan_id>/file', methods=['GET'])
def get_scan_file(scan_id):
    print(scan_id )
    # Retrieve the filename from the MySQL table
    with connection.cursor() as cursor:
        cursor.execute("SELECT 3dfile_path FROM scans_history WHERE scan_id = %s", (scan_id))
        filename = cursor.fetchone()

    if filename:
        # Construct the file path
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_path = filename[0]
        print(file_path)

        # Return the file
        return send_file(file_path)
    else:
        return jsonify({'error': 'File not found'}), 404
@app.route('/scans', methods=['GET'])
def get_all_scans():
    # Retrieve query parameters for pagination, filtering, and sorting
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    sort_by = request.args.get('sort_by', 'scan_id')
    sort_order = request.args.get('sort_order', 'asc')
    filter_by = request.args.get('filter_by')
    filter_value = request.args.get('filter_value')

    # Calculate the offset based on the current page and items per page
    offset = (page - 1) * per_page

    # Construct the SQL query with pagination, filtering, and sorting
    query = "SELECT * FROM scans_history"
    if filter_by and filter_value:
        query += f" WHERE {filter_by} = '{filter_value}'"
    query += f" ORDER BY {sort_by} {sort_order.upper()} LIMIT {offset}, {per_page}"

    # Execute the query
    with connection.cursor() as cursor:
        cursor.execute(query)
        data = cursor.fetchall()
    
    # Convert each row into a dictionary with column names as keys
    rows_as_dicts = []
    column_names = [col[0] for col in cursor.description]
    for row in data:
        row_dict = {col_name: value for col_name, value in zip(column_names, row)}
        rows_as_dicts.append(row_dict)

    # Return the list of dictionaries as JSON
    return jsonify(rows_as_dicts)


# Route to retrieve a single scan by ID
@app.route('/scans/<int:id>', methods=['GET'])
def get_scan_by_id(id):
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM scans_history WHERE scan_id  = %s", (id,))
        data = cursor.fetchone()

    if data:
        # Convert the row into a dictionary with column names as keys
        column_names = [col[0] for col in cursor.description]
        row_dict = {col_name: value for col_name, value in zip(column_names, data)}
        return jsonify(row_dict)
    else:
        return jsonify({"message": "Scan not found"}), 404

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the file with a temporary filename
        temp_filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(file_path)

        # Insert file information into MySQL table
        with connection.cursor() as cursor:
            # Retrieve the auto-increment ID generated by MySQL
            cursor.execute("SELECT scan_id FROM scans_history ORDER BY scan_id DESC LIMIT 1")
            scan_id = cursor.fetchone()[0] + 1

            print("Latest auto-increment ID:", scan_id)

        # Rename the file with the auto-increment ID
        new_filename = f"{scan_id}_scan_object_file.obj"
        new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        os.rename(file_path, new_file_path)

    # Access the array data from the request
    data_array = json.loads(request.form.getlist('parameters')[0])
    print("Received Data Array:", round(data_array['LG'],4) )
    
    # Load your 3D model
    mesh = trimesh.load_mesh(new_file_path)
    
    metric = data_array['metric'] 
    print(metric)
    # Three main points on the leg

    LG =round(data_array['LG'],4)          #crotch_point
    LE =round(data_array['LE'],4)          #knee_point
    LA =round(data_array['LA'],4)          #ankle_point

    # find all other points
    LF = (LG + LE) / 2.0 

    # Calculate the intervals
    interval1 = (LE - LA) / 4
    interval2 = 2 * interval1
    interval3 = 3 * interval1

    # Calculate the three points
    LB = LA + interval1
    LC = LA + interval2
    LD = LA + interval3


    LG_origin, LG_normal =[0, LG, 0], [0, 1, 0]
    LF_origin, LF_normal =[0, LF, 0], [0, 1, 0]
    LE_origin, LE_normal =[0, LE, 0], [0, 1, 0]
    LD_origin, LD_normal =[0, LD, 0], [0, 1, 0]
    LC_origin, LC_normal =[0, LC, 0], [0, 1, 0]
    LB_origin, LB_normal =[0, LB, 0], [0, 1, 0]
    LA_origin, LA_normal =[0, LA, 0], [0, 1, 0]

    # Get all the circular lengths
    CG =get_circular_length(LG,mesh,LG_origin,LG_normal)
    CF =get_circular_length(LF,mesh,LF_origin,LF_normal)
    CE =get_circular_length(LE,mesh,LE_origin,LE_normal)
    CD =get_circular_length(LD,mesh,LD_origin,LD_normal)
    CC =get_circular_length(LC,mesh,LC_origin,LC_normal)
    CB =get_circular_length(LB,mesh,LB_origin,LB_normal)
    CA =get_circular_length(LA,mesh,LA_origin,LA_normal)
   
   # Prepare the measurement data in JSON format
    measurement_data = {
        'metric' : metric,
        'image_path' : 'file.filename/myfile.jpg',
        'patient_name' : 'Ali Hamza',
        '3dfile_path': new_file_path,
        'doctor_name' : 'Shahrose Khan',
        'CG' : convert_measure(CG, metric),
        'CF' : convert_measure(CF, metric),
        'CE' : convert_measure(CE, metric),
        'CD' : convert_measure(CD, metric),
        'CC' : convert_measure(CC, metric),
        'CB' : convert_measure(CB, metric),
        'CA' : convert_measure(CA, metric),
        'LG' : convert_measure(LG, metric),
        'LF' : convert_measure(LF, metric),
        'LE' : convert_measure(LE, metric),
        'LD' : convert_measure(LD, metric),
        'LC' : convert_measure(LC, metric),
        'LB' : convert_measure(LB, metric),
        'LA' : convert_measure(LA, metric),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Generate column names and values for the SQL query
    columns = ', '.join(measurement_data.keys())
    values = ', '.join(['%s' for _ in range(len(measurement_data))])


    # Insert the measurements into the database
    with connection.cursor() as cursor:
        sql_query = f"INSERT INTO scans_history ({columns}) VALUES ({values})"  # Generate the SQL query
        query_values = [measurement_data[key] for key in measurement_data]  # Extract values from the dictionary in the same order as columns
        cursor.execute(sql_query, query_values)
        connection.commit()
  
    # Return the measurements as JSON response
    return jsonify(measurement_data)

@app.route('/test')
def test():
    return jsonify({'success': 'true'})

if __name__ == '__main__':
    app.run(debug=True)