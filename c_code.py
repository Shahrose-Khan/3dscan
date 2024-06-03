from flask import Flask, render_template, request, jsonify, send_file
import pymysql
from functools import wraps
from flask_bcrypt import Bcrypt
import jwt
from datetime import datetime, timedelta, UTC
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

def get_db_connection():
    return pymysql.connect(host='127.0.0.1',
                           user='root',
                           password='root',
                           db='3dscan_db')

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

def slice_3d_model(plane_origin,plane_normal,mesh):
  # Perform the slicing using the slice_plane method
  slice_mesh = mesh.slice_plane(plane_origin, plane_normal)

  slice_mesh = trimesh.Trimesh(vertices=slice_mesh.vertices,
                             faces=slice_mesh.faces)
  return slice_mesh


def get_circular_length(height,mesh):
    plane_origin, plane_normal =[0, height, 0], [0, 1, 0]

    slice_mesh = mesh.section(plane_origin=plane_origin,plane_normal=plane_normal)
    
    if slice_mesh is None or len(slice_mesh.vertices) < 3:
        return 0
    
    try:
        s_mesh = trimesh.Trimesh(vertices=slice_mesh.vertices)
        hull_vertices = s_mesh.convex_hull.vertices

        if len(hull_vertices) < 3:
            return 0
        #Get convex hull as a new Trimesh object
        hull_mesh = s_mesh.convex_hull

        # Create a closed path using the vertices
        points = hull_mesh.vertices[:, [0, 2]]
        # Compute the convex hull
        hull = ConvexHull(points)

        # Get the perimeter (length) of the convex hull
        perimeter_length = hull.area

        return perimeter_length
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0

def get_circular_length_z_axis(mesh, plane_origin, plane_normal):
    # Perform the slicing using the section method
    slice_mesh = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    
    if slice_mesh is None or len(slice_mesh.vertices) < 3:
        return 0

    try:
        s_mesh = trimesh.Trimesh(vertices=slice_mesh.vertices)
        hull_vertices = s_mesh.convex_hull.vertices
        
        if len(hull_vertices) < 3:
            return 0
        
        # Get convex hull as a new Trimesh object
        hull_mesh = s_mesh.convex_hull
        
        # Create a closed path using the vertices
        points = hull_mesh.vertices[:, [0, 1]]
        
        # Compute the convex hull
        hull = ConvexHull(points)
        
        # Get the perimeter (length) of the convex hull
        perimeter_length = hull.area
        
        return perimeter_length
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0
def get_CA(foot):
    # Define the number of slices
    num_slices = 100

    # Define the z-axis range based on the bounding box of the mesh
    z_min, z_max = foot.bounds[0][2], foot.bounds[1][2]
    z_values = np.linspace(z_min, z_max, num_slices)

    CA = 0
    i = 0
    # Iterate over each z-value and slice the mesh
    for z_value in z_values:
        # Define the slicing plane
        plane_origin = [0, 0, z_value]
        plane_normal = [0, 0, 1]
        i = i + 1
        
        circumference=get_circular_length_z_axis(foot,plane_origin,plane_normal)

        if i % 70 == 0:
            CA = circumference
     
    return CA


def get_CY(foot):
    num_slices = 100

    # Define the z-axis range based on the bounding box of the mesh
    z_min, z_max = foot.bounds[0][2], foot.bounds[1][2]
    z_values = np.linspace(z_min, z_max, num_slices)

    CYs = []
    i =0
    # Iterate over each z-value and slice the mesh
    for z_value in z_values:
        i = i + 1
        # Define the slicing plane
        plane_origin = [0, 0.04, z_value-0.05]
        plane_normal = [0, -0.5, 0.6]

        circumference=get_circular_length_z_axis(foot,plane_origin,plane_normal)
        CYs.append([i,circumference])

    max_CY = 0
    for i,CY in CYs:
        if(CY > max_CY):
            max_CY = CY

    max_CY = max_CY * 1.10

    return max_CY

def get_ankle_length(mesh, num_points=30):
    height = get_mesh_height(mesh)
    min_length = float('inf')
    min_height = None
    circular_lengths = []
    
    # Calculate the height of each segment
    segment_height = height / (6 * num_points)
    
    for i in range(num_points):
        height = i * segment_height + 0.04
        circular_length = get_circular_length(height, mesh)
        # circular_lengths.append([convert_measure(circular_length,'in'),convert_measure(height,'in')])
        # Update minimum length and height
        if circular_length < min_length:
            min_length = circular_length
            min_height = height
    
    return min_length, min_height

def generate_measure_response(db_record):
    if db_record:
        # Define default values for null entries
        default_value = 0.0

        # Prepare the measurement data in JSON format
        right_measurement_data = {
            'CG': db_record.get('R_CG', default_value),
            'CF': db_record.get('R_CF', default_value),
            'CE': db_record.get('R_CE', default_value),
            'CD': db_record.get('R_CD', default_value),
            'CC': db_record.get('R_CC', default_value),
            'CB1': db_record.get('R_CB1', default_value),
            'CB': db_record.get('R_CB', default_value),
            'CY': db_record.get('R_CY', default_value),
            'CA': db_record.get('R_CA', default_value),
            'LG': db_record.get('R_LG', default_value),
            'LF': db_record.get('R_LF', default_value),
            'LE': db_record.get('R_LE', default_value),
            'LD': db_record.get('R_LD', default_value),
            'LC': db_record.get('R_LC', default_value),
            'LB1': db_record.get('R_LB1', default_value),
            'LB': db_record.get('R_LB', default_value),
            'LA': db_record.get('R_LA', default_value)  # Using get() method to handle KeyError
        }

        left_measurement_data = {
            'CG': db_record.get('L_CG', default_value),
            'CF': db_record.get('L_CF', default_value),
            'CE': db_record.get('L_CE', default_value),
            'CD': db_record.get('L_CD', default_value),
            'CC': db_record.get('L_CC', default_value),
            'CB1': db_record.get('L_CB1', default_value),
            'CB': db_record.get('L_CB', default_value),
            'CY': db_record.get('L_CY', default_value),
            'CA': db_record.get('L_CA', default_value),
            'LG': db_record.get('L_LG', default_value),
            'LF': db_record.get('L_LF', default_value),
            'LE': db_record.get('L_LE', default_value),
            'LD': db_record.get('L_LD', default_value),
            'LC': db_record.get('L_LC', default_value),
            'LB1': db_record.get('L_LB1', default_value),
            'LB': db_record.get('L_LB', default_value),
            'LA': db_record.get('L_LA', default_value)  # Using get() method to handle KeyError
        }

        measurement_data = {
            'scan_id': db_record.get('scan_id', 0),
            'right': right_measurement_data,
            'left': left_measurement_data,
            'metric': db_record.get('metric', ''),
            'patient_name': db_record.get('patient_name', ''),
            'doctor_name': db_record.get('doctor_name', ''),
            'original_file_path': db_record.get('original_file_path', ''),
            'processed_file_path': db_record.get('processed_file_path', ''),
            'left_file_path': db_record.get('left_file_path', ''),
            'right_file_path': db_record.get('right_file_path', ''),
            'timestamp': db_record.get('timestamp', ''),
            'is_manual': db_record.get('is_manual', 0),
        }
        return measurement_data
    else:
        return None




# Decorator function to verify JWT token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Check if 'Authorization' header is present in the request
        if 'Authorization' in request.headers:
            # Split the header value by whitespace
            auth_header_parts = request.headers['Authorization'].split()
            # Check if the split result contains expected parts
            if len(auth_header_parts) == 2:
                # Extract token from the split result
                token = auth_header_parts[1]

        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        try:
            # Decode the token
            data = jwt.decode(token, 'your_secret_key', algorithms=['HS256'])
            # Add the decoded token data to the request object for further processing
            request.current_user = data
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401

        # Call the original function with the current user
        return f(*args, **kwargs)

    return decorated




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
bcrypt = Bcrypt()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scans/<int:scan_id>/file', methods=['GET'])
def get_scan_file(scan_id):
    print(scan_id )
    # Retrieve the filename from the MySQL table
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT 3dfile_path FROM scans_history WHERE scan_id = %s", (scan_id))
        filename = cursor.fetchone()
    
    connection.close()

    if filename:
        # Construct the file path
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_path = filename[0]
        # Return the file
        return send_file(file_path)
    else:
        return jsonify({'error': 'File not found'}), 404
 
@app.route('/scans/weekly-report', methods=['GET'])
def get_weekly_report():
    # Calculate the start and end dates for the last seven days
    end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    start_date = end_date - timedelta(days=6)

    # Construct the SQL query to get the weekly report
    query = f"""
    SELECT day_of_week, scan_count
    FROM (
        SELECT DAYNAME(timestamp) AS day_of_week,
               COUNT(*) AS scan_count
        FROM scans_history
        WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
        GROUP BY day_of_week
    ) AS subquery
    ORDER BY FIELD(day_of_week, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday');
    """


    print(query)
    connection = get_db_connection()
    # Execute the query
    with connection.cursor() as cursor:
        cursor.execute(query)
        weekly_report = cursor.fetchall()

    connection.close()

    # Create a dictionary to hold the weekly report
    weekly_report_dict = {
        "Mon": 0,
        "Tue": 0,
        "Wed": 0,
        "Thu": 0,
        "Fri": 0,
        "Sat": 0,
        "Sun": 0
    }

    # Populate the dictionary with fetched data
    for entry in weekly_report:
        day_of_week = entry[0][:3]  # Take the first three characters for short day name
        scan_count = entry[1]        # Access the second element of the tuple
        weekly_report_dict[day_of_week] = scan_count

    return jsonify(weekly_report_dict)

@app.route('/scanner-doctors', methods=['GET'])
def get_verified_scanner_doctors():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT doctor_id, username, full_name FROM doctors WHERE is_verified = 1 AND is_Scanner = 1")
            doctors = cursor.fetchall()

        # Convert the list of tuples to a list of dictionaries
        doctors_list = [{'doctor_id': doctor[0], 'username': doctor[1], 'full_name': doctor[2]} for doctor in doctors]

        return jsonify(doctors_list), 200
    except Exception as e:
        return jsonify({'error': 'An error occurred while fetching doctors data'}), 500




@app.route('/scans', methods=['GET'])
# @token_required
def get_all_scans():
    # Retrieve query parameters for pagination, filtering, and sorting
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 100))
    sort_by = request.args.get('sort_by', 'scan_id')
    sort_order = request.args.get('sort_order', 'desc')
    filter_by = request.args.get('filter_by')
    filter_value = request.args.get('filter_value')

    # Calculate the offset based on the current page and items per page
    offset = (page - 1) * per_page

    # Construct the SQL query with pagination, filtering, and sorting
    query = "SELECT * FROM scans_history"
    if filter_by and filter_value:
        query += f" WHERE {filter_by} = '{filter_value}'"
    query += f" ORDER BY {sort_by} {sort_order.upper()} LIMIT {offset}, {per_page}"

    connection = get_db_connection()
    # Execute the query
    with connection.cursor() as cursor:
        cursor.execute(query)
        data = cursor.fetchall()

    connection.close()
    
    # Convert each row into a dictionary with column names as keys
    db_records = []
    column_names = [col[0] for col in cursor.description]
    for row in data:
        db_record = {col_name: value for col_name, value in zip(column_names, row)}
        db_records.append(generate_measure_response(db_record))

    # Return the list of dictionaries as JSON
    return jsonify(db_records)

def update_measurement(scan_id, leg, part,  new_length, new_circumference):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        if leg == 'left':
            length_column = 'L_L' + part 
            circumference_column = 'L_C' + part 
        elif leg == 'right':
            length_column = 'R_L' + part 
            circumference_column = 'R_C' + part 
            
        else:
            return jsonify({"message": "Invalid leg specified"}), 400
        
        # Check if part is 'A' or 'Y'
        if part in ['A', 'Y']:
            # Update only the circumference column
            cursor.execute(f"UPDATE scans_history SET `{circumference_column}` = %s WHERE scan_id = %s", (new_circumference, scan_id))
        else:
            cursor.execute(f"UPDATE scans_history SET `{length_column}` = %s, `{circumference_column}` = %s WHERE scan_id = %s", (new_length, new_circumference, scan_id))
        
        connection.commit()
        connection.close()
        return jsonify({"message": f"Successfully updated measurements for scan {scan_id}"}), 200


@app.route('/scans/update/<int:scan_id>', methods=['POST'])
def update_measurement_route(scan_id):
    data=json.loads(request.form.getlist('parameters')[0])
    if 'leg' not in data or 'part' not in data or 'new_length' not in data or 'new_circumference' not in data:
        return jsonify({"message": "Missing required fields in request body"}), 400

    leg = data['leg']
    part = data['part']
    new_length = data['new_length']
    new_circumference = data['new_circumference']

    return update_measurement(scan_id, leg, part, new_length,new_circumference)

# Route to retrieve a single scan by ID
@app.route('/scans/<int:scan_id>', methods=['GET'])
def get_scan_by_id(scan_id):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM scans_history WHERE scan_id  = %s", (scan_id,))
        data = cursor.fetchone()
    connection.close()

    if data:
        # Convert the row into a dictionary with column names as keys
        column_names = [col[0] for col in cursor.description]
        db_record = {col_name: value for col_name, value in zip(column_names, data)}

        return jsonify(generate_measure_response(db_record))
    else:
        return jsonify({"message": "Scan not found"}), 404
    
@app.route('/measure/point/<int:scan_id>', methods=['GET'])
def get_measure_at_height_scan_by_id(scan_id):

    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT `left_file_path`, `right_file_path` FROM scans_history WHERE scan_id  = %s", (scan_id,))
        data = cursor.fetchone()

    connection.close()
  
    if data:
        left_leg_file_path = data[0]
        right_leg_file_path = data[1]

        left_leg_mesh = trimesh.load_mesh(left_leg_file_path)
        right_leg_mesh = trimesh.load_mesh(right_leg_file_path)


        height = float(request.args.get('height', 0))
        metric = request.args.get('metric', 'in')


        # Prepare the measurement data in JSON format
        measurement_data = {
            'left_measure' : convert_measure(get_circular_length(height,left_leg_mesh), metric),
            'right_measure' : convert_measure(get_circular_length(height,right_leg_mesh), metric),
            'height' : convert_measure(height, metric),
            'metric' : metric,
            'scan_id': scan_id
        }

        # Return the measurements as JSON response
        return jsonify(measurement_data)
    else:
        return jsonify({"message": "Scan not found"}), 404


@app.route('/measure/<int:scan_id>', methods=['GET'])
def get_measure_scan_by_id(scan_id):

    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT `left_file_path`, `right_file_path` FROM scans_history WHERE scan_id  = %s", (scan_id,))
        data = cursor.fetchone()

    connection.close()
  
    if data:
        left_leg_file_path = data[0]
        right_leg_file_path = data[1]

        left_leg_mesh = trimesh.load_mesh(left_leg_file_path)
        right_leg_mesh = trimesh.load_mesh(right_leg_file_path)

        left_leg_height = get_mesh_height(left_leg_mesh) 
        right_leg_height = get_mesh_height(right_leg_mesh) 

        plane_origin=[0, 0.2, 0]
        plane_normal=[0, -1, 0]


        
        left_foot_mesh =slice_3d_model(plane_origin,plane_normal,left_leg_mesh)
        right_foot_mesh =slice_3d_model(plane_origin,plane_normal,right_leg_mesh)



        ankle_length, ankle_height = get_ankle_length(right_leg_mesh)
        res = [convert_measure(ankle_length, 'in'),convert_measure(ankle_height, 'in')]
        # res = get_ankle_length(right_leg_mesh)

        doctor_name = request.args.get('doctorName', '')  
        patient_name = request.args.get('patientName', '')  
        metric = request.args.get('metric', 'in')

        print(metric)
        # Three main points on the leg

        LG =right_leg_height  * 0.9                          #crotch_point
        LE =(LG + ankle_height)/2          #knee_point
        LB = ankle_height                                #ankle_point

        # find all other points
        LF = (LG + LE) / 2.0 

        # Calculate the intervals
        interval1 = (LE - LB) / 4
        interval2 = 2 * interval1
        interval3 = 3 * interval1

        # Calculate the three points
        LB1 = LB + interval1
        LC = LB + interval2
        LD = LB + interval3

        # Prepare the measurement data in JSON format
        right_measurement_data = {
            'R_CG' : convert_measure(get_circular_length(LG,right_leg_mesh), metric),
            'R_CF' : convert_measure(get_circular_length(LF,right_leg_mesh), metric),
            'R_CE' : convert_measure(get_circular_length(LE,right_leg_mesh), metric),
            'R_CD' : convert_measure(get_circular_length(LD,right_leg_mesh), metric),
            'R_CC' : convert_measure(get_circular_length(LC,right_leg_mesh), metric),
            'R_CB1' : convert_measure(get_circular_length(LB1,right_leg_mesh), metric),
            'R_CB' : convert_measure(get_circular_length(LB,right_leg_mesh), metric),
            'R_CY' : convert_measure(get_CY(right_foot_mesh), metric),
            'R_CA' : convert_measure(get_CA(right_foot_mesh), metric),
            'R_LG' : convert_measure(LG, metric),
            'R_LF' : convert_measure(LF, metric),
            'R_LE' : convert_measure(LE, metric),
            'R_LD' : convert_measure(LD, metric),
            'R_LC' : convert_measure(LC, metric),
            'R_LB1' : convert_measure(LB1, metric),
            'R_LB' : convert_measure(LB, metric)
        }

        ankle_length, ankle_height = get_ankle_length(left_leg_mesh)
        res = [convert_measure(ankle_length, 'in'),convert_measure(ankle_height, 'in')]
        # res = get_ankle_length(left_leg_mesh)

        # Three main points on the leg

        LG =left_leg_height * 0.9                            #crotch_point
        LE =(LG + ankle_height)/2          #knee_point
        LB = ankle_height                                #ankle_point

        # find all other points
        LF = (LG + LE) / 2.0 

        # Calculate the intervals
        interval1 = (LE - LB) / 4
        interval2 = 2 * interval1
        interval3 = 3 * interval1

        # Calculate the three points
        LB1 = LB + interval1
        LC = LB + interval2
        LD = LB + interval3

        # Prepare the measurement data in JSON format
        left_measurement_data = {
            'L_CG' : convert_measure(get_circular_length(LG,left_leg_mesh), metric),
            'L_CF' : convert_measure(get_circular_length(LF,left_leg_mesh), metric),
            'L_CE' : convert_measure(get_circular_length(LE,left_leg_mesh), metric),
            'L_CD' : convert_measure(get_circular_length(LD,left_leg_mesh), metric),
            'L_CC' : convert_measure(get_circular_length(LC,left_leg_mesh), metric),
            'L_CB1' : convert_measure(get_circular_length(LB1,left_leg_mesh), metric),
            'L_CB' : convert_measure(get_circular_length(LB,left_leg_mesh), metric),
            'L_CY' : convert_measure(get_CY(left_foot_mesh), metric),
            'L_CA' : convert_measure(get_CA(left_foot_mesh), metric),
            'L_LG' : convert_measure(LG, metric),
            'L_LF' : convert_measure(LF, metric),
            'L_LE' : convert_measure(LE, metric),
            'L_LD' : convert_measure(LD, metric),
            'L_LC' : convert_measure(LC, metric),
            'L_LB1' : convert_measure(LB1, metric),
            'L_LB' : convert_measure(LB, metric)
        }

        measurement_data = {
            **left_measurement_data,
            **right_measurement_data,
            'metric' : metric,
            'doctor_name' : doctor_name, 
            'patient_name' : patient_name , 


        }

        # Generate column names and values for the SQL query
        update_columns = ', '.join([f"{key} = %s" for key in measurement_data.keys()])  # Include all keys from measurement_data
        update_values = [measurement_data[key] for key in measurement_data]  # Extract values from the dictionary in the same order as columns

        # Add scan_id to the values list
        update_values.append(scan_id)

        # Construct the SQL query to update the record based on scan_id
        sql_query = f"UPDATE scans_history SET {update_columns} WHERE scan_id = %s"

        connection = get_db_connection()
        # Execute the SQL query to update the record
        with connection.cursor() as cursor:
            cursor.execute(sql_query, update_values)
            connection.commit()
        
        connection.close()


    
        # Return the measurements as JSON response
        # return jsonify(measurement_data)
        return get_scan_by_id(scan_id)
    
        # #resoponse construction
        # response = {
        #     'left_leg_height':left_leg_height,
        #     'right_leg_height': right_leg_height,
        #     'res': res
        # }
        # return jsonify(response)
    else:
        return jsonify({"message": "Scan not found"}), 404

@app.route('/process', methods=['POST'])
def process_file():
    BASE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'scans')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the file with a temporary filename
        temp_filename = file.filename
        file_path = os.path.join(BASE_PATH, 'original',temp_filename)
        file.save(file_path)

        connection = get_db_connection()
        # Insert file information into MySQL table
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLE STATUS LIKE 'scans_history'")
            table_status = cursor.fetchone()
            scan_id = table_status[10]
        
        connection.close()

        # Rename the file with the auto-increment ID
        new_filename = f"{scan_id}_scan_file.obj"
        new_file_path = os.path.join(BASE_PATH, 'original', new_filename)
        os.rename(file_path, new_file_path)

    # Access the array data from the request
    if 'parameters' in request.form:
        data_array = json.loads(request.form.getlist('parameters')[0])
        height = data_array['maxHeight']
    else:
        height = 0.9
    
    # Load your 3D model
    full_mesh = trimesh.load_mesh(new_file_path)
    # Define a slicing plane along a custom direction
    plane_origin=[0, height, 0]
    plane_normal=[0,-1, 0]

    mesh = slice_3d_model(plane_origin,plane_normal,full_mesh)
    processed_file_path = os.path.join(BASE_PATH, 'processed', f"{scan_id}_scan_file_processed.obj")
    mesh.export(processed_file_path) 

    plane_origin=[0, 0, 0]
    plane_normal=[1, 0, 0]
    left_part =slice_3d_model(plane_origin,plane_normal,mesh)
 
    plane_normal=[-1,0, 0]
    right_part =slice_3d_model(plane_origin,plane_normal,mesh)

    left_file_path = os.path.join(BASE_PATH, 'left', f"{scan_id}_scan_file_left.obj")
    right_file_path = os.path.join(BASE_PATH, 'right', f"{scan_id}_scan_file_right.obj")
    
    left_part.export(left_file_path) 
    right_part.export(right_file_path) 

     
   # Prepare the measurement data in JSON format
    measurement_data = {
        'original_file_path': new_file_path,
        'processed_file_path' : processed_file_path,
        'left_file_path': left_file_path,
        'right_file_path': right_file_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Generate column names and values for the SQL query
    columns = ', '.join(measurement_data.keys())
    values = ', '.join(['%s' for _ in range(len(measurement_data))])


    connection = get_db_connection()
    # Insert the measurements into the database
    with connection.cursor() as cursor:
        sql_query = f"INSERT INTO scans_history ({columns}) VALUES ({values})"  # Generate the SQL query
        query_values = [measurement_data[key] for key in measurement_data]  # Extract values from the dictionary in the same order as columns
        cursor.execute(sql_query, query_values)
        connection.commit()
    
    connection.close()

    measurement_data['scan_id'] = scan_id
    # Return the measurements as JSON response
    return jsonify(measurement_data)


@app.route('/upload', methods=['POST'])
def upload_file():
    BASE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'scans')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the file with a temporary filename
        temp_filename = file.filename
        file_path = os.path.join(BASE_PATH, 'original',temp_filename)
        file.save(file_path)

        connection = get_db_connection()
        # Insert file information into MySQL table
        with connection.cursor() as cursor:
            # Retrieve the auto-increment ID generated by MySQL
            cursor.execute("SELECT scan_id FROM scans_history ORDER BY scan_id DESC LIMIT 1")
            scan_id = cursor.fetchone()[0] + 1

        connection.close()

        # Rename the file with the auto-increment ID
        new_filename = f"{scan_id}_scan_file.obj"
        new_file_path = os.path.join(BASE_PATH, 'original', new_filename)
        os.rename(file_path, new_file_path)

    # Access the array data from the request
    data_array = json.loads(request.form.getlist('parameters')[0])
    print("Received Data Array:", round(data_array['LG'],4) )
    
    # Load your 3D model
    full_mesh = trimesh.load_mesh(new_file_path)

    height=1
    # Define a slicing plane along a custom direction
    plane_origin=[0, height, 0]
    plane_normal=[0,-1, 0]


    mesh = slice_3d_model(plane_origin,plane_normal,full_mesh)
    processed_file_path = os.path.join(BASE_PATH, 'processed', f"{scan_id}_scan_file_processed.obj")
    mesh.export(processed_file_path) 

    plane_origin=[0, 0, 0]
    plane_normal=[1, 0, 0]
    left_part =slice_3d_model(plane_origin,plane_normal,mesh)

    plane_normal=[-1,0, 0]
    right_part =slice_3d_model(plane_origin,plane_normal,mesh)

    left_file_path = os.path.join(BASE_PATH, 'left', f"{scan_id}_scan_file_left.obj")
    right_file_path = os.path.join(BASE_PATH, 'right', f"{scan_id}_scan_file_right.obj")
    
    left_part.export(left_file_path) 
    right_part.export(right_file_path) 


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
    CG =get_circular_length(LG,left_part)
    CF =get_circular_length(LF,left_part)
    CE =get_circular_length(LE,left_part)
    CD =get_circular_length(LD,left_part)
    CC =get_circular_length(LC,left_part)
    CB =get_circular_length(LB,left_part)
    CA =get_circular_length(LA,left_part)
   
   # Prepare the measurement data in JSON format
    measurement_data = {
        'metric' : metric,
        'image_path' : 'file.filename/myfile.jpg',
        'patient_name' : 'Ali Hamza',
        'original_file_path': new_file_path,
        'processed_file_path' : processed_file_path,
        'left_file_path': left_file_path,
        'right_file_path': right_file_path,
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


    connection = get_db_connection()
    # Insert the measurements into the database
    with connection.cursor() as cursor:
        sql_query = f"INSERT INTO scans_history ({columns}) VALUES ({values})"  # Generate the SQL query
        query_values = [measurement_data[key] for key in measurement_data]  # Extract values from the dictionary in the same order as columns
        cursor.execute(sql_query, query_values)
        connection.commit()
    
    connection.close()
  
    # Return the measurements as JSON response
    return jsonify(measurement_data)

def get_mesh_height(mesh):
    """
    Calculate the height of a mesh object.
    
    Parameters:
        mesh (trimesh.Trimesh): The mesh object.
        
    Returns:
        float: The height of the mesh.
    """
    # Compute the bounding box of the mesh
    bbox = mesh.bounds
    
    # Extract the height from the bounding box
    height = bbox[1, 1] - bbox[0, 1]
    
    return height

# Function to generate JWT token
def generate_token(user_id, user_type):
    try:
        payload = {
            'user_id': user_id,
            'user_type': user_type,
            'exp': datetime.now(UTC) + timedelta(minutes=30)  # Token expires in 30 minutes
        }
        token = jwt.encode(payload, 'your_secret_key', algorithm='HS256')
        return token
    except Exception as e:
        return str(e)
    
# Login route
@app.route('/login', methods=['POST'])
def login():
    connection = None
    try:
        username = request.json['username']
        password = request.json['password']
        is_doctor = request.json['isDoctor']  # True if user is a doctor, False if user is a patient

        connection = get_db_connection()
        cursor = connection.cursor()

        if is_doctor:
            # Check if the doctor exists
            cursor.execute("SELECT * FROM doctors WHERE username = %s", (username,))
            doctor = cursor.fetchone()
            if doctor:
                # Verify password
                if bcrypt.check_password_hash(doctor[2], password):
                    # Generate access token
                    access_token = generate_token(doctor[0], 'doctor')
                    return jsonify({'access_token': access_token}), 200
                else:
                    return jsonify({'message': 'Invalid username or password'}), 401
            else:
                return jsonify({'message': 'Doctor not found'}), 404
        else:
            # Check if the patient exists
            cursor.execute("SELECT * FROM patients WHERE username = %s", (username,))
            patient = cursor.fetchone()
            if patient:
                # Verify password
                if bcrypt.check_password_hash(patient[2], password):
                    # Generate access token
                    access_token = generate_token(patient[0], 'patient')
                    return jsonify({'access_token': access_token}), 200
                else:
                    return jsonify({'message': 'Invalid username or password'}), 401
            else:
                return jsonify({'message': 'Patient not found'}), 404
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        if connection:
            connection.close()
# Signup route
@app.route('/signup', methods=['POST'])
def signup():
    connection = None  # Initialize connection variable
    try:
        username = request.json['username']
        password = request.json['password']
        email = request.json['email']
        full_name = request.json['full_name']
        is_doctor = request.json['isDoctor']  # True if user is a doctor, False if user is a patient

        connection = get_db_connection()
        cursor = connection.cursor()

        if is_doctor:
            # Check if the username or email already exists
            cursor.execute("SELECT * FROM doctors WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()
            if existing_user:
                return jsonify({'message': 'Username or email already exists'}), 400

            # Hash the password
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

            # Insert the new doctor into the database
            cursor.execute("INSERT INTO doctors (username, password, email, full_name, is_admin) VALUES (%s, %s, %s, %s, %s)",
                           (username, hashed_password, email, full_name, False))
            connection.commit()

            return jsonify({'message': 'Doctor registered successfully'}), 201
        else:
            # Check if the username or email already exists
            cursor.execute("SELECT * FROM patients WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()
            if existing_user:
                return jsonify({'message': 'Username or email already exists'}), 400

            # Hash the password
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

            # Insert the new patient into the database
            cursor.execute("INSERT INTO patients (username, password, email, full_name) VALUES (%s, %s, %s, %s)",
                           (username, hashed_password, email, full_name))
            connection.commit()

            return jsonify({'message': 'Patient registered successfully'}), 201
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        if connection:
            connection.close()

@app.route('/insert_scan_data', methods=['POST'])
def insert_scan_data():
    try:
        # Parse JSON data from the request
        data = request.json
        
        # Extract relevant information from the JSON data
        doctor_name = data.get('doctor_name')
        patient_name = data.get('patient_name')
        is_manual = data.get('is_manual')
        metric = data.get('metric')
        left_data = data.get('left', {})
        right_data = data.get('right', {})
        
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Construct SQL query to insert the data into the database
            sql_query = """
                INSERT INTO scans_history (doctor_name, patient_name, is_manual, metric, L_CA, L_CY, L_CB, L_CB1, L_CC, L_CD, L_CE, L_CF, L_CG, L_LB, L_LB1, L_LC, L_LD, L_LE, L_LF, L_LG, R_CA, R_CY, R_CB, R_CB1, R_CC, R_CD, R_CE, R_CF, R_CG, R_LB, R_LB1, R_LC, R_LD, R_LE, R_LF, R_LG)
                VALUES (%s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Execute the SQL query to insert the data
            cursor.execute(sql_query, (doctor_name, patient_name, is_manual, metric, 
                                       left_data.get('CA'), left_data.get('CY'), left_data.get('CB'), left_data.get('CB1'),
                                       left_data.get('CC'), left_data.get('CD'), left_data.get('CE'), left_data.get('CF'),
                                       left_data.get('CG'), left_data.get('LB'), left_data.get('LB1'), left_data.get('LC'),
                                       left_data.get('LD'), left_data.get('LE'), left_data.get('LF'), left_data.get('LG'),
                                       right_data.get('CA'), right_data.get('CY'), right_data.get('CB'), right_data.get('CB1'),
                                       right_data.get('CC'), right_data.get('CD'), right_data.get('CE'), right_data.get('CF'),
                                       right_data.get('CG'), right_data.get('LB'), right_data.get('LB1'), right_data.get('LC'),
                                       right_data.get('LD'), right_data.get('LE'), right_data.get('LF'), right_data.get('LG')))
            connection.commit()
        
        return jsonify({'message': 'Data inserted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()


@app.route('/update_scan_data', methods=['POST'])
def update_scan_data():
    try:
        # Parse JSON data from the request
        data = request.json
        
        # Extract relevant information from the JSON data
        scan_id = data.get('scan_id')
        doctor_name = data.get('doctor_name')
        patient_name = data.get('patient_name')
        is_manual = data.get('is_manual')
        metric = data.get('metric')
        left_data = data.get('left', {})
        right_data = data.get('right', {})
        
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Construct SQL query to update the data in the database
            sql_query = """
                UPDATE scans_history
                SET doctor_name = %s, patient_name = %s, is_manual = %s, metric = %s,
                    L_CA = %s, L_CY = %s, L_CB = %s, L_CB1 = %s, L_CC = %s, L_CD = %s, L_CE = %s, L_CF = %s, L_CG = %s, L_LB = %s, L_LB1 = %s, L_LC = %s, L_LD = %s, L_LE = %s, L_LF = %s, L_LG = %s,
                    R_CA = %s, R_CY = %s, R_CB = %s, R_CB1 = %s, R_CC = %s, R_CD = %s, R_CE = %s, R_CF = %s, R_CG = %s, R_LB = %s, R_LB1 = %s, R_LC = %s, R_LD = %s, R_LE = %s, R_LF = %s, R_LG = %s
                WHERE scan_id = %s
            """
            
            # Execute the SQL query to update the data
            cursor.execute(sql_query, (doctor_name, patient_name, is_manual, metric, 
                                       left_data.get('CA'), left_data.get('CY'), left_data.get('CB'), left_data.get('CB1'),
                                       left_data.get('CC'), left_data.get('CD'), left_data.get('CE'), left_data.get('CF'),
                                       left_data.get('CG'), left_data.get('LB'), left_data.get('LB1'), left_data.get('LC'),
                                       left_data.get('LD'), left_data.get('LE'), left_data.get('LF'), left_data.get('LG'),
                                       right_data.get('CA'), right_data.get('CY'), right_data.get('CB'), right_data.get('CB1'),
                                       right_data.get('CC'), right_data.get('CD'), right_data.get('CE'), right_data.get('CF'),
                                       right_data.get('CG'), right_data.get('LB'), right_data.get('LB1'), right_data.get('LC'),
                                       right_data.get('LD'), right_data.get('LE'), right_data.get('LF'), right_data.get('LG'),
                                       scan_id))
            connection.commit()
        
        return jsonify({'message': 'Data updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()

if __name__ == '__main__':
    app.run(port=5500,debug=True)
    # app.run(host='127.0.0.1', port=5000)


