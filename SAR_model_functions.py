"""
This module contains a suite of utility functions specifically designed for enhancing geographic data manipulation and analysis in search and rescue (SAR) operations.
It offers methods for terrain encoding based on RGB values, integrating trails, buildings, and railways into terrain data, and generating slope matrices for terrain assessment.
The module also provides functions for simulating lost person movement in the terrain based on terrain score matrices
and creating map layers for most efficent search areas based on the results of the simulation.

Note:
The module is designed to be used in the SAR (Search and Rescue) application SR Kart created during bachelor's thesis at UiT The Arctic University of Norway.
The tools provided here are intended to support the development of a web-based application for search and rescue operations in Norway,
focusing on the use of geospatial data to optimize search strategies and improve the efficiency of rescue missions.

Dependencies:
    - utility (custom module)
    - numpy
    - geopandas
    - matplotlib
    

Author: Martin Riksheim
Date: 2024
"""

try:
    from utility import (
        matrix_value_padding,
        normalize_component,
        compute_concave_hull_from_points,
        create_polygon_map_overlay,
        plot_branching_result,
        normalize_array,
        create_square_polygon,
        transform_coordinates_to_utm
    )
except ImportError:
    from .utility import (
        matrix_value_padding,
        normalize_component,
        compute_concave_hull_from_points,
        create_polygon_map_overlay,
        plot_branching_result,
        normalize_array,
        create_square_polygon,
        transform_coordinates_to_utm
    )
import numpy as np
import geopandas as gpd
import math
import random
import time




# Function to encode terrain types based on RGB values
def terrain_encoding(terrain_type_encoding, terrain_rgb_values, terrain_filepath, folder="output/array/", search_id=0):
    """
    Encode terrain types based on RGB values.

    Parameters:
    terrain_type_encoding (dict): A dictionary mapping terrain names to their corresponding encodings.
    terrain_rgb_values (dict): A dictionary mapping terrain names to their corresponding RGB values.
    terrain_filepath (str): The file path of the terrain RGB data.
    folder (str, optional): The folder path to save the encoded terrain data. Defaults to "output/array/".
    search_id (int, optional): The search ID for the encoded terrain data. Defaults to 0.

    Returns:
    None
    """

    try:
        terrain_rgb_data_3d = np.load(f'{terrain_filepath}')
    except:
        print(f'No terrain data found in {terrain_filepath}')
        return

    print(f'Color analysis terrain encoding started...')
    print(f'{terrain_rgb_data_3d.shape[1]}x{terrain_rgb_data_3d.shape[2]} pixels to process.')

    # Transpose the 3D array so that color channels are last
    terrain_rgb_data_2d = terrain_rgb_data_3d.transpose((1, 2, 0))
   
    # Flatten the 2D RGB array to a 1D array for efficient processing
    flat_rgb_data = terrain_rgb_data_2d.reshape(-1, 3)

    # Create unique RGB combinations and corresponding indices
    unique_colors, inverse_indices = np.unique(flat_rgb_data, axis=0, return_inverse=True)
    
    # Create a mapping array for encodings
    color_to_encoding_map = np.full(unique_colors.shape[0], terrain_type_encoding["Ukjent"], dtype=float)
    for terrain_name, rgb in terrain_rgb_values.items():
        # Compare unique colors directly without structured arrays
        matches = np.all(unique_colors == np.array(rgb), axis=1)
        matched_indices = np.where(matches)[0]
        if matched_indices.size > 0:
            color_to_encoding_map[matched_indices[0]] = terrain_type_encoding[terrain_name]
    
    # Use the inverse indices to map the encodings back to the original image shape
    encoded_terrain = color_to_encoding_map[inverse_indices].reshape(terrain_rgb_data_2d.shape[0], terrain_rgb_data_2d.shape[1])
    
    # Pad Y axis for swamps since they are horizontal lines seperated by white lines
    offset = [(1, 0), (-1, 0)]
    swamp_val = terrain_type_encoding["Myr"]
    matrix_value_padding(encoded_terrain, swamp_val, 1, offset)
    
    encoded_terrain[0][0] = 1 # To assure that the min value is atleast 1 for better visual when plotting 

    # pad all roads and trails
    encoded_terrain = matrix_value_padding(encoded_terrain, terrain_type_encoding["Sti og vei"], 1)

    np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', encoded_terrain)
    print(f'Encoded terrain data np array saved to {folder}id{search_id}_terrain_type_matrix.npy')






def add_trails_data_to_terrain(terrain_data, trail_files=[],folder="output/array/", trail_value=1, search_id=0):
        """
    Adds trail data to the terrain data array.

    Parameters:
    - terrain_data (numpy.ndarray): The terrain data array.
    - trail_files (list): List of trail files to be added.
    - folder (str): The folder path where the trail files are located.
    - trail_value (int): The value to assign to the trail data in the terrain array.
    - search_id (int): The search ID used for saving the terrain data.

    Returns:
    None
    """
        trails_added = False

        for trail_file in trail_files:
            try:
                trails = np.load(f'{folder}{trail_file}')
                terrain_data[trails == 1] = trail_value
                trails_added = True
            except:
                print(f'No trail data found in {folder}{trail_file}')
        
        if trails_added:
            try:
                np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', terrain_data)
                print(f'Trails added to terrain data and saved to {folder}id{search_id}_terrain_type_matrix.npy')
            except PermissionError: # if file is open
                while True:
                    try:
                        np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', terrain_data)
                        print(f'Trails added to terrain data and saved to {folder}id{search_id}_terrain_type_matrix.npy')
                        break
                    except PermissionError:
                        time.sleep(1)
        else:
            print(f'No trail data added to terrain data.')
            #np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', terrain_data)


def add_building_data_to_terrain(terrain_data, building_files=[],folder="output/array/", building_value=0, search_id=0):
        buildings_added = False
        for building_file in building_files:
            try:
                buildings = np.load(f'{folder}{building_file}')
                terrain_data[buildings == 1] = building_value
                buildings_added = True
            except:
                print(f'No building data found in {folder}{building_file}')
        
        if buildings_added:
            try:
                np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', terrain_data)
                print(f'Buildings added to terrain data and saved to {folder}id{search_id}_terrain_type_matrix.npy')
            except PermissionError: # if file is open
                while True:
                    try:
                        np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', terrain_data)
                        print(f'Buildings added to terrain data and saved to {folder}id{search_id}_terrain_type_matrix.npy')
                        break
                    except PermissionError:
                        time.sleep(1)
        else:
            print(f'No building data added to terrain data.')
            


def add_railway_data_to_terrain(terrain_data, railway_files=[],folder="output/array/", railway_value=0.8, search_id=0):
        railway_added = False
        for railway_file in railway_files:
            try:
                railway = np.load(f'{folder}{railway_file}')
                terrain_data[railway == 1] = railway_value
                railway_added = True
            except:
                print(f'No railway data found in {folder}{railway_file}')
        
        if railway_added:
            try:
                np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', terrain_data)
                print(f'Railways added to terrain data and saved to {folder}id{search_id}_terrain_type_matrix.npy')
            except PermissionError: # if file is open
                while True:
                    try:
                        np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', terrain_data)
                        print(f'Railways added to terrain data and saved to {folder}id{search_id}_terrain_type_matrix.npy')
                        break
                    except PermissionError:
                        time.sleep(1)
        else:
            print(f'No railway data added to terrain data.')
                  

    

def calc_steepness(height_matrix):
    # Gradients along both axes
    grad_x, grad_y = np.gradient(height_matrix)
    # Steepness for each square (pythagorean)
    steepness_matrix = np.sqrt(grad_x**2 + grad_y**2)
    return steepness_matrix



def combine_terrain_type_and_slope(terrain_type_matrix, slope_matrix, method="multiply", folder="output/array/", search_id=0):
    """
    Combines the terrain type matrix and slope matrix to create a combined terrain score matrix.
    
    Parameters:
    terrain_type_matrix (numpy.ndarray): The matrix representing the terrain type.
    slope_matrix (numpy.ndarray): The matrix representing the slope.
    method (str, optional): The method used to combine the matrices. Defaults to "multiply".
    folder (str, optional): The folder path to save the output matrix. Defaults to "output/array/".
    search_id (int, optional): The search ID used in the output matrix file name. Defaults to 0.
    
    Returns:
    numpy.ndarray: The combined terrain score matrix.
    """

    slope_matrix[terrain_type_matrix == 1] = 1  # Exclude elevation data from trails
    combined_matrix = combine_matrixes(terrain_type_matrix, slope_matrix, method)
    print(f"Combined terrain score matrix saved to: {folder}{search_id}_terrain_score_matrix.npy")
    np.save(f'{folder}id{search_id}_terrain_score_matrix.npy', combined_matrix)
    return combined_matrix


def combine_matrixes(terrain_type, slope, method="mean"):
    if terrain_type.shape != slope.shape:
        print("Matrixes are not the same size")
        return
    if method == "mean":
        return (terrain_type + slope) / 2
    elif method == "multiply":
        return terrain_type * slope
    elif method == "square":
        return (terrain_type * slope) ** 2
    elif method == "cube":
        return (terrain_type * slope) ** 3


def calc_travel_distance(matrix, energy, center, end_x, end_y, step_limit=9999):
    """ Only used for staright line traversal simulation. Not used in the final implementation.
    """
    x0, y0 = center
    x1, y1 = end_x, end_y
    dx = abs(x1-x0)
    dy = abs(y1-y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    energy_used = 0
    steps = 0

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            energy_used += 1 / matrix[x][y]
            steps += 1
            err -= dy
            if err < 0:
                y += sy
                err += dx
                energy_used += 0.5 / matrix[x][y]
            x += sx
            if energy_used > energy or steps > step_limit:
                break
    else:
        err = dy / 2.0
        while y != y1:
            energy_used += 1 / matrix[x][y]
            steps += 1
            err -= dx
            if err < 0:
                x += sx
                err += dy
                energy_used += 0.5 / matrix[x][y]
            y += sy
            if energy_used > energy or steps > step_limit:
                break

    energy_left = energy - energy_used
    return (x, y), energy_left    # end point



def movement(matrix, start_idx, move_dir, obstacle_threshold):
    """ Returns new position and energy cost for the movement.
        Used in branching_movement.
    """
    
    x0, y0 = start_idx
    sx, sy = move_dir

    x = x0 + sx
    y = y0 + sy

    try:
        # obstacle, step back
        if matrix[y, x] <= obstacle_threshold:
            return (x0, y0), 2 / matrix[y0, x0]
        
        # out of bounds in negative direction, step back and end
        if x < 0 or y < 0:
            return (x0, y0), float('inf')
            
        # diagonal
        if abs(sx) == abs(sy):
            energy_cost = (math.sqrt(2) / matrix[y, x])

        # straight
        elif abs(sx) != abs(sy):
            energy_cost = (1 / matrix[y, x])

        # somewhere else...
        else:
            print("how did you get here?")
            energy_cost = float('inf')

    except:
        return (x0, y0), float('inf')
    
    return (x, y), energy_cost



def branching_movement(matrix, start_idx, move_dir, initial_move_resource, move_resource, sets, ring_25, ring_50, terrain_change_threshold, random_branching_chance, obstacle_threshold):
    """
    Perform branching movement algorithm in a matrix.
    Saves the coordinates in different sets based on energy left.

    Args:
        matrix (numpy.ndarray): The matrix representing the terrain.
        start_idx (tuple): The starting index for the movement.
        move_dir (tuple): The initial movement direction.
        initial_move_resource (int): The initial movement resource.
        move_resource (int): The current movement resource.
        sets (tuple): A tuple containing the sets from the main function.
        ring_25 (int): The threshold for the 25% ring.
        ring_50 (int): The threshold for the 50% ring.
        terrain_change_threshold (int): The threshold for significant terrain change.
        random_branching_chance (float): The chance of random branching.
        obstacle_threshold (int): The threshold for obstacle detection.

    Returns:
        None
    """
    
    start_time = time.perf_counter()
    time_limit = 300

    # referenced sets from main function
    green, yellow, red, branch_log, last_cutoff = sets

    # Initialize stack with starting point
    stack = [(start_idx, move_dir, move_resource)]
    
    # Thresholds for to group into different rings
    ring_25_threshold = initial_move_resource - ring_25
    ring_50_threshold = initial_move_resource - ring_50
    cutoff_threshold = initial_move_resource - (ring_50 + ((initial_move_resource - ring_50) / 2))



    while stack:
        current_idx, move_dir, move_resource = stack.pop()
        x, y = current_idx

        if move_resource <= 0:
            red.add(current_idx)
            continue
        
        # Stack limit
        if len(stack) > 50:
            continue

        # Movement
        (new_x, new_y), movement_cost = movement(matrix, current_idx, move_dir, obstacle_threshold)
        energy_left = move_resource - movement_cost
        
        # Save coords for relevant rings based on energy left
        if move_resource > ring_25_threshold:
            green.add((new_x, new_y))
        elif move_resource > ring_50_threshold:
            yellow.add((new_x, new_y))
        else:
            red.add((new_x, new_y))
            if move_resource > cutoff_threshold:
                last_cutoff.add((new_x, new_y))
        

        # Branches kill offs (Early end if low chance of improvement)
        if move_resource < initial_move_resource - (ring_25 * 1.1) and (new_x, new_y) in green:
            continue
        if move_resource < initial_move_resource - (ring_50 * 1.1) and (new_x, new_y) in yellow:
            continue
        if move_resource < initial_move_resource - ((ring_50 + ((initial_move_resource - ring_50)/2)) * 1.1) and (new_x, new_y) in last_cutoff:
            continue
       
        
        if energy_left > 0:
            # Check terrain change
            if 0 <= new_y < len(matrix) and 0 <= new_x < len(matrix[0]):
                terrain_change = matrix[new_y, new_x] - matrix[y, x]
            else:
                terrain_change = 10    # Out of bounds
            
            # Branching multipliers based on energy left and distance params
            if energy_left > initial_move_resource - ring_25:
                random_branch_multiplier = 60 if ring_25 <= 200 else 20
            elif energy_left > initial_move_resource - ring_50:
                random_branch_multiplier = 40 if ring_50 <= 500 else 20
            elif energy_left > initial_move_resource - (ring_50 + ((initial_move_resource - ring_50)/2)):
                random_branch_multiplier = 10
            else:
                random_branch_multiplier = 20
            
            # Branching if terrain change is significant
            if abs(terrain_change) > terrain_change_threshold and (new_x, new_y) not in branch_log:
                stack.append(((new_x, new_y), move_dir, energy_left))
                
                sx, sy = move_dir
                if terrain_change < 0:  # worse terrain
                    new_directions = [
                        (normalize_component(sx + sy), normalize_component(sy - sx)),  # right diagonal
                        (normalize_component(sx - sy), normalize_component(sy + sx))   # left diagonal
                    ]
                else:  # better terrain
                    if random.random() > 0.5:
                        new_directions = [
                            (sy, -sx),  # right
                            (-sy, sx)   # left
                        ]
                    else:
                        new_directions = [
                            (normalize_component(sx + sy), normalize_component(sy - sx)),  # right diagonal
                            (normalize_component(sx - sy), normalize_component(sy + sx))   # left diagonal
                        ]
                
                for new_dir in new_directions:
                    branch_log.add((new_x + new_dir[0], new_y + new_dir[1]))
                    stack.append(((new_x, new_y), new_dir, energy_left))
            
            # Random branching
            elif (random_branch_multiplier * random_branching_chance) > random.randint(1, 10000) and (new_x, new_y) not in branch_log:
                
                # Time out check here to avoid checking for each loop
                if time.perf_counter() - start_time > time_limit:
                    print(f'Time limit reached.')
                    break
                
                sx, sy = move_dir
                new_directions = [
                    (sx, sy),  # forward
                    (sy, -sx),  # right
                    (-sy, sx),  # left
                    (normalize_component(sx + sy), normalize_component(sy - sx)),  # right diagonal
                    (normalize_component(sx - sy), normalize_component(sy + sx))   # left diagonal
                ]
                
                for new_dir in new_directions:
                    branch_log.add((new_x + new_dir[0], new_y + new_dir[1]))
                    stack.append(((new_x, new_y), new_dir, energy_left))
            
            # Position not changed, meaning obstacle. Randomly change direction
            elif (x, y) == (new_x, new_y):
                sx, sy = move_dir
                random_num = random.randint(1, 4)
                new_dir = {
                    1: (sy, -sx),
                    2: (-sy, sx),
                    3: (normalize_component(sx + sy), normalize_component(sy - sx)),
                    4: (normalize_component(sx - sy), normalize_component(sy + sx))
                }[random_num]


                stack.append(((new_x, new_y), new_dir, energy_left))
            
            # Continue in the same direction
            else:
                stack.append(((new_x, new_y), move_dir, energy_left))


        # Energy depleted, end branch      
        else:
            red.add((new_x, new_y))



def calculate_map_extension(max_distance, square_radius):
    """
    Calculates the map extension based on the maximum distance and square radius.

    Parameters:
    - max_distance (float): The maximum distance.
    - square_radius (float): The square radius.

    Returns:
    - map_extension (int): The calculated map extension.
    """
    
    map_square = 2*square_radius
    map_size = 2*max_distance
    print(f'{map_size=}')

    if map_size <= map_square:
        map_extension = 0
    elif map_size <= 3*map_square:
        map_extension = 1
    elif map_size <= 5*map_square:
        map_extension = 2
    elif map_size <= 7*map_square:
        map_extension = 3
    elif map_size <= 9*map_square:
        map_extension = 4
    elif map_size <= 11*map_square:
        map_extension = 5
    elif map_size <= 13*map_square:
        map_extension = 6
    else:
        map_extension = 7
    
    return map_extension


def create_slope_matrix(height_matrix, norm_cap, square_factor, folder, search_id):
    """
    Create a slope matrix based on the given height matrix.

    Args:
        height_matrix (numpy.ndarray): The input height matrix.
        norm_cap (float): The maximum value for the slope matrix.
        square_factor (float): The factor to apply to the inverse slope matrix.
        folder (str): The folder path to save the slope matrix.
        search_id (int): The search ID for the slope matrix.

    Returns:
        numpy.ndarray: The inverse slope matrix.

    """
    # Calculate the slope matrix
    slope_matrix = calc_steepness(height_matrix) 

    # Cap the slope matrix values
    slope_matrix[slope_matrix > norm_cap] = norm_cap

    # Normalize the slope matrix
    norm_slope_matrix = normalize_array(slope_matrix, norm_cap)

    # Inverse the slope matrix
    inv_slope_matrix = 1 - norm_slope_matrix
    inv_slope_matrix = inv_slope_matrix ** square_factor # more seperation between values

    np.save(f'{folder}id{search_id}_slope_matrix.npy', inv_slope_matrix)
    return inv_slope_matrix



def branching_simulation(terrain_score_matrix, d25, d50, d75, config):
    """
    Simulates branching movement in the terrain score matrix based on given parameters.

    Args:
        terrain_score_matrix (numpy.ndarray): Matrix representing the terrain score.
        d25 (float): Distance value for the 25th percentile.
        d50 (float): Distance value for the 50th percentile.
        d75 (float): Distance value for the 75th percentile.
        config (object): Configuration object containing various parameters.

    Returns:
        tuple: A tuple containing three numpy arrays representing the red points, yellow points, and green points.

    """

    logfile = config.LOG_FILE
    worse_terrain_threshold = config.TERRAIN_CHANGE_THRESHOLD
    obstacle_threshold = config.OBSTACLE_THRESHOLD
    random_branching_chance = config.RANDOM_FACTOR
    
    radius = terrain_score_matrix.shape[0] / 2
    center = (int(radius), int(radius))

    ring_25 = (d25 / config.REDUCTION_FACTOR) * config.RANGE_FACTOR
    ring_50 = (d50 / config.REDUCTION_FACTOR) * config.RANGE_FACTOR
    ring_75 = (d75 / config.REDUCTION_FACTOR) * config.RANGE_FACTOR

    print(f'{d25=}, {d50=}, {d75=}')
    print(f'{ring_25=}, {ring_50=}, {ring_75=}')

    max_distance = ring_75
    
    # Initialize sets for storing coordinates in different groups
    green_coords = set()
    yellow_coords = set()
    red_coords = set()
    branches_log = set()
    last_cutoff = set()
    sets = (green_coords, yellow_coords, red_coords, branches_log, last_cutoff)

    # For logging progress
    step_percentage = (100 / (config.ITERATIONS * 8))
    completion_percentage = 0
    with open(logfile, 'ab') as f:
        text = f'{completion_percentage}%'
        f.write(text.encode())

    time_limit = 300
    sim_start_time = time.perf_counter()

    # Calls branching_movement for each direction (8 directions in total) per iteration
    for n in range(config.ITERATIONS):
        curr_dir = 1
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                move_direction = (i, j)
                if move_direction != (0, 0):
                    print(f'Iteration {n+1}/{config.ITERATIONS}, Direction {curr_dir}/{8}')
                    curr_dir += 1
                    if time.perf_counter() - sim_start_time > time_limit:
                        print(f'Time limit reached.')
                        break

                    branches_log = set() # reset branches
                    sets = (green_coords, yellow_coords, red_coords, branches_log, last_cutoff)
                    branching_movement(terrain_score_matrix, (center[0], center[1]), move_direction, max_distance, max_distance, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance, obstacle_threshold)
                    
                    # For logging progress
                    log_step_back = len(f'{completion_percentage:.0f}%')
                    completion_percentage += step_percentage
                    with open(logfile, 'rb+') as f:
                        f.seek(0, 2)  # Move to the end of the file
                        file_size = f.tell()
                        text = f'{completion_percentage:.0f}%'
                        f.seek(max(0, file_size - log_step_back), 0)  # Move pointer back
                        f.write(text.encode())  # Write the updated percentage
    
    # For logging progress
    with open(logfile, 'rb+') as f:
        f.seek(0, 2)  # Move to the end of the file
        file_size = f.tell()
        text = f'. 100%\n'
        log_step_back = len(text)
        f.seek(max(0, file_size - log_step_back), 0)  # Move pointer back
        f.write(text.encode())  # Write the final percentage

    # convert sets to np arrays
    red_points = np.array(list(red_coords)[::10])
    yellow_points = np.array(list(yellow_coords)[::3])
    green_points = np.array(list(green_coords)[::2])

    return red_points, yellow_points, green_points


def create_map_layer(terrain_score_matrix, start_coords, red_points, yellow_points, green_points, folder, search_id, config):
    """
    Create map layers based on the terrain score matrix and point data.

    Args:
        terrain_score_matrix (numpy.ndarray): The terrain score matrix.
        start_coords (tuple): The starting coordinates.
        red_points (list): List of red points.
        yellow_points (list): List of yellow points.
        green_points (list): List of green points.
        folder (str): The folder to save the map layers.
        search_id (str): The search ID.
        config (Config): The configuration object.

    Returns:
        tuple: A tuple containing the map layers for 25%, 50%, and 75% confidence levels.
    """

    # Compute concave hulls from the red, yellow, and green points
    concave_hull_r = compute_concave_hull_from_points(red_points, config.HULL_ALPHA)
    concave_hull_y = compute_concave_hull_from_points(yellow_points, config.HULL_ALPHA)
    concave_hull_g = compute_concave_hull_from_points(green_points, config.HULL_ALPHA)
    
    # Save branching result as png for logging
    plot_branching_result(terrain_score_matrix, concave_hull_r, concave_hull_y, concave_hull_g, config, save=True)

    # Create map overlays for the 25%, 50%, and 75% confidence levels with CRS EPSG:4326
    polygon_75 = create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_r, color="red", output_crs="EPSG:4326", folder=folder, reduction_factor=config.REDUCTION_FACTOR, search_id=search_id)
    polygon_50 = create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_y, color="yellow", output_crs="EPSG:4326", folder=folder, reduction_factor=config.REDUCTION_FACTOR, search_id=search_id)
    polygon_25 = create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_g, color="green", output_crs="EPSG:4326", folder=folder, reduction_factor=config.REDUCTION_FACTOR, search_id=search_id)

    return polygon_25, polygon_50, polygon_75
    
    


def create_search_sectors_with_polygons(matrix, coords, hull_polygon, sector_size=50_000, reduction_factor=5, output_crs="EPSG:25833", folder='output/overlays/', search_id=0):
    """
    Create search sectors with polygons based on a matrix, coordinates, and a hull polygon.

    Args:
        matrix (numpy.ndarray): The matrix used to determine the size of the search sectors.
        coords (tuple): The coordinates (latitude, longitude) used as the center of the search sectors.
        hull_polygon (shapely.geometry.Polygon): The hull polygon used to determine the intersection with the search sectors.
        sector_size (int, optional): The size of each search sector in square meters. Defaults to 50,000.
        reduction_factor (int, optional): The reduction factor used to calculate the map diameter. Defaults to 5.
        output_crs (str, optional): The output coordinate reference system (CRS) for the search sectors. Defaults to "EPSG:25833".
        folder (str, optional): The folder path to save the search sector polygons. Defaults to 'output/overlays/'.
        search_id (int, optional): The ID of the search. Defaults to 0.

    Returns:
        list: A list of intersected sector polygons.
    """
    
    height, width = matrix.shape
    map_diameter = height * reduction_factor  # Meters
    distance_per_index = map_diameter / height  # Meters per index in the matrix
    lat, lng = coords
    center_x, center_y = transform_coordinates_to_utm(lat, lng)

    sector_side_length_m = np.sqrt(sector_size)
    sector_side_length_idx = int(sector_side_length_m / reduction_factor)

    sector_polygons = []

    for i in range(0, height, sector_side_length_idx):
        for j in range(0, width, sector_side_length_idx):
            sector_center_x_idx = j + sector_side_length_idx / 2
            sector_center_y_idx = i + sector_side_length_idx / 2

            x_meter = (sector_center_x_idx - width / 2) * distance_per_index
            y_meter = ((height - sector_center_y_idx) - height / 2) * distance_per_index

            sector_center_x_geo, sector_center_y_geo = center_x + x_meter, center_y + y_meter

            square_polygon = create_square_polygon(sector_center_x_geo, sector_center_y_geo, sector_side_length_m)

            # Check for intersection with the freeform search area polygon
            if hull_polygon.intersects(square_polygon):
                sector_polygons.append(square_polygon)

    base_crs = 'EPSG:25833'
    gdf = gpd.GeoDataFrame(index=range(len(sector_polygons)), crs=base_crs, geometry=sector_polygons)
    gdf.to_crs(output_crs, inplace=True)

    intersected_sector_polygons = []

    for idx, polygon in enumerate(gdf.geometry):
        gdf_single = gpd.GeoDataFrame(index=[0], crs=output_crs, geometry=[polygon])
        intersected_sector_polygons.append(gdf_single.geometry[0])
        gdf_single.to_file(f'{folder}id{search_id}_sector_{idx}_EPSG{output_crs[5:]}.geojson', driver='GeoJSON')

    return intersected_sector_polygons