from sarModel.modelFunctions.utility import *
from sarModel.modelFunctions.constants import *
import numpy as np
import math
import random
import time
from scipy.ndimage import maximum_filter

#############################################################
""" For debugging branching simulation """
cnt = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
cnt_kill_1 = 0
cnt_kill_2 = 0
cnt_kill_3 = 0
def debug_stats_print():
    global cnt, cnt2, cnt3, cnt4, cnt_kill_1, cnt_kill_2, cnt_kill_3
    print(f'1:{cnt_kill_1},2:{cnt_kill_2},3:{cnt_kill_3}, Total:{cnt_kill_1+cnt_kill_2+cnt_kill_3} Killed branches')

    print(f'{cnt} Steps')
    print(f'{cnt2} Branching, random')
    print(f'{cnt4} Branching, worse terrain')
    print(f'{cnt3} Direction change, obstacle')
#############################################################


def terrain_encoding(terrain_type_encoding, terrain_rgb_values, terrain_filepath, folder="output/array/", search_id=0):
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
    np.save(f'{folder}id{search_id}_terrain_data_encoded.npy', encoded_terrain)
    print(f'Encoded terrain data np array saved to {folder}terrain_data_encoded.npy')






def add_trails_data_to_terrain(terrain_data, trail_files=[],folder="output/array/", trail_value=1, search_id=0):
        trails_added = False
        
        terrain_data = matrix_value_padding(terrain_data, trail_value,  3)   # road padding

        for trail_file in trail_files:
            try:
                trails = np.load(f'{folder}{trail_file}')

                trails = matrix_value_padding(trails, trail_value, 5) # trail padding

                terrain_data[trails == 1] = trail_value
                trails_added = True
            except:
                print(f'No trail data found in {folder}{trail_file}')
        
        if trails_added:
            #neighbour_offset = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            #steps = 8
            #terrain_data = set_neighbours(terrain_data, trail_value, neighbour_offset, 3)   # trail and road padding
            np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', terrain_data)
            print(f'Trails added to terrain data and saved to {folder}id{search_id}_terrain_type_matrix.npy')
        else:
            print(f'No trail data added to terrain data.')
            #np.save(f'{folder}terrain_type_matrix.npy', terrain_data)


def add_building_data_to_terrain(terrain_data, building_files=[],folder="output/array/", building_value=0, search_id=0):
        buildings_added = False
        for building_file in building_files:
            try:
                buildings = np.load(f'{folder}{building_file}')

                buildings = matrix_value_padding(buildings, 1, 2) # building padding

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
                        np.save(f'{folder}terrain_type_matrix.npy', terrain_data)
                        print(f'Buildings added to terrain data and saved to {folder}terrain_type_matrix.npy')
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

                railway = matrix_value_padding(railway, 1, 5) # padding

                terrain_data[railway == 1] = railway_value
                railway_added = True
            except:
                print(f'No railway data found in {folder}{railway_file}')
        
        if railway_added:
            try:
                np.save(f'{folder}id{search_id}_terrain_type_matrix.npy', terrain_data)
                print(f'Railways added to terrain data and saved to {folder}terrain_type_matrix.npy')
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
    # gradients along both axes
    grad_x, grad_y = np.gradient(height_matrix)
    # steepness for each square (pythagorean)
    steepness_matrix = np.sqrt(grad_x**2 + grad_y**2)
    return steepness_matrix


def reduce_resolution(data, factor=10, method="mean"):
    if method == "mean":
        return data.reshape(data.shape[0]//factor, factor, data.shape[1]//factor, factor).mean(axis=1).mean(axis=2)
    elif method == "max":
        return data.reshape(data.shape[0]//factor, factor, data.shape[1]//factor, factor).max(axis=1).max(axis=2)
    elif method == "min":
        return data.reshape(data.shape[0]//factor, factor, data.shape[1]//factor, factor).mean(axis=1).min(axis=2)



def combine_terrain_type_and_slope(terrain_type, slope, method="mean", filter_size=3, folder="output/array/", search_id=0):
    terrain_type[terrain_type <= 0.05] = 0
    slope[slope <= 0.1] = 0
    combined_matrix = combine_matrixes(terrain_type, slope, method)
    combined_matrix_filtered = maximum_filter(combined_matrix, size=filter_size)
    print(f"Combined terrain score matrix saved to: {folder}{search_id}_terrain_score_matrix.npy")
    np.save(f'{folder}id{search_id}_terrain_score_matrix.npy', combined_matrix_filtered)
    return combined_matrix_filtered


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
    """ Only used for staright line traversal
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





    
    

def movement(matrix, start_idx, move_dir):
    """ Returns new position and energy cost for the movement.
        Used in branching_movement.
    """
    global cnt
    cnt += 1
    x0, y0 = start_idx
    sx, sy = move_dir

    x = x0 + sx
    y = y0 + sy

    try:
        # obstacle, step back
        if matrix[y, x] <= 0:
            return (x0, y0), 2 / matrix[y0, x0]
        
        # out of bounds in negative direction, step back and end
        if x < 0 or y < 0:
            return (x0, y0), float('inf')
            
        # diagonal
        if abs(sx) == abs(sy):
            energy_cost = math.sqrt(2) / matrix[y, x]

        # straight
        elif abs(sx) != abs(sy):
            energy_cost = 1 / matrix[y, x]

        # somewhere else...
        else:
            print("how did you get here?")
            energy_cost = float('inf')

    except:
        return (x0, y0), float('inf')
    
    return (x, y), energy_cost


def branching_movement(matrix, current_idx, move_dir, initial_move_resource, move_resource, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance, step_count):
    global cnt, cnt2, cnt3, cnt4, cnt_kill_1, cnt_kill_2, cnt_kill_3
    

    green, yellow, red, branches, last_cutoff = sets   # refrenced sets from main function
    x, y = current_idx
    if move_resource <= 0:
        red.add(current_idx) # red coords
        return

    # movement
    (new_x, new_y), movement_cost = movement(matrix, current_idx, move_dir)
    step_count += 1
    energy_left = move_resource - movement_cost

    # save green and yellow coords if energy dips threshold
    if move_resource > initial_move_resource*(100-ring_25)/100:
        if energy_left < initial_move_resource*(100-ring_25)/100:
            green.add((new_x, new_y))
    elif move_resource > initial_move_resource*(100-ring_50)/100:
        if energy_left < initial_move_resource*(100-ring_50)/100:
            if (new_x, new_y) in green:
                cnt_kill_1 += 1
                return
            else:
                yellow.add((new_x, new_y))


    elif move_resource > initial_move_resource*((100-ring_50)/2)/100:
        #print("Start of last cutoff")
        if energy_left < initial_move_resource*((100-ring_50)/2)/100:
            if (new_x, new_y) in green or (new_x, new_y) in yellow:
                #print(f'Killing branch: {new_x, new_y}')
                cnt_kill_2 += 1
                return
            else:
                #print(f'Adding to last cutoff: {new_x, new_y}')
                last_cutoff.add((new_x, new_y))


    
    # branches kill offs
    if move_resource < initial_move_resource*(100-((ring_50+ring_25)/2))/100:
        if (new_x, new_y) in green:
            cnt_kill_1 += 1
            return

    if move_resource < initial_move_resource*(100-ring_50)/100:
        if (new_x, new_y) in yellow:
            cnt_kill_2 += 1
            return

    if move_resource < initial_move_resource*((100-ring_50)/2)/100:
        if (new_x, new_y) in last_cutoff:
            cnt_kill_3 += 1
            return
    #print(f'{step_count}/{initial_move_resource*2} steps')
    if step_count > initial_move_resource*2:
        print(f'Step limit reached: {step_count}')
        red.add((new_x, new_y)) # red coords
        return



    if energy_left > 0:    
        
        try:
            terrain_change = matrix[new_y, new_x] - matrix[y, x]
        except:
            terrain_change = 10  # prob out of bounds

        # Branching if terrain change is significant worse
        if terrain_change < -worse_terrain_threshold and (new_x, new_y) not in branches:
            cnt4 += 1
            # continue in the same direction
            branching_movement(matrix, (new_x, new_y), move_dir, initial_move_resource, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance, step_count)
            
            sx, sy = move_dir   # current direction
            # Branches in 4 directions. diagonals in same direction and sideways.
            new_directions = [
                (sy,-sx),   # right
                (-sy,sx),   # left
                #(normalize_component(sx+sy),normalize_component(sy-sx)), # right diagonal
                #(normalize_component(sx-sy),normalize_component(sy+sx)) # left diagonal
            ]

            for new_dir in new_directions:
                branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                branching_movement(matrix, (new_x, new_y), new_dir, initial_move_resource, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance, step_count)

            

        # Random branching. Branches in all 8 directions
        elif random.randint(1,10000) <= random_branching_chance and (new_x, new_y) not in branches:
            cnt2 += 1
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    new_dir = (i, j)
                    if new_dir != (0, 0):
                        branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                        branching_movement(matrix, (new_x, new_y), new_dir, initial_move_resource, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance, step_count)
        
        # Change direction if obstacle, randomly choose right or left
        elif (x, y) == (new_x, new_y):
            cnt3 += 1
            if random.randint(0,1) == 1:
                new_dir = (move_dir[1], -move_dir[0]) # right
            else:
                new_dir = (-move_dir[1], move_dir[0])  # left
            branching_movement(matrix, (new_x, new_y), new_dir, initial_move_resource, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance, step_count)

        

        # Continue in the same direction 
        else:               
            branching_movement(matrix, (new_x, new_y), move_dir, initial_move_resource, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance, step_count)
        
    else:
        # Energy depleted, stop recursion and save end point
        red.add((new_x, new_y)) # red coords
        

def calculate_map_extension(max_distance, square_radius, extra_space):
    print(square_radius, max_distance, extra_space)
    map_square = square_radius * 2
    map_size = max_distance * (1 + extra_space)     # 50% extra space

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
    else:
        map_extension = 5
    
    return map_extension


def create_slope_matrix(height_matrix, norm_cap, folder, search_id=0):
    slope_matrix = calc_steepness(height_matrix)
    slope_matrix[slope_matrix > norm_cap] = norm_cap
    norm_slope_matrix = normalize_array(slope_matrix, norm_cap)
    inv_slope_matrix = 1 - norm_slope_matrix
    inv_slope_matrix = inv_slope_matrix ** 2 # Squaring whole array to make the difference between high and low values bigger
    np.save(f'{folder}id{search_id}_slope_matrix.npy', inv_slope_matrix)
    return inv_slope_matrix

def branching_simulation(terrain_score_matrix, search_id=0):
    ring_25 = BranchingConfig.D25.value
    ring_50 = BranchingConfig.D50.value
    worse_terrain_threshold = BranchingConfig.WORSE_TERRAIN.value
    random_branching_chance = BranchingConfig.RANDOM_FACTOR.value
    
    radius = terrain_score_matrix.shape[0] / 2
    center = (int(radius), int(radius))
    max_distance = radius * BranchingConfig.RANGE_FACTOR.value
    max_energy = max_distance
    
    green_coords = set()
    yellow_coords = set()
    red_coords = set()
    branches = set()
    last_cutoff = set()
    sets = (green_coords, yellow_coords, red_coords, branches, last_cutoff)

    start_time = time.perf_counter()
    for n in range(BranchingConfig.ITERATIONS.value):
        curr_dir = 1
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                move_direction = (i, j)
                if move_direction != (0, 0):
                    print(f'Iteration {n+1}/{BranchingConfig.ITERATIONS.value}, Direction {curr_dir}/{8}')
                    curr_dir += 1
                    branches = set() # reset branches
                    sets = (green_coords, yellow_coords, red_coords, branches, last_cutoff)
                    branching_movement(terrain_score_matrix, (center[0], center[1]), move_direction, max_energy, max_energy, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance, 0)
                    
    end_time = time.perf_counter()


    print(f'Params: {BranchingConfig.ITERATIONS.value} iter, {BranchingConfig.RANGE_FACTOR.value} b_range, {worse_terrain_threshold} terrain_thrshld, {random_branching_chance} random_chance')
    print(f"Branching simulation took {end_time - start_time} seconds")
    print(f'Unique paths simulated: {len(red_coords)}')  # number of endpoints/paths
    debug_stats_print()

    # convert sets to np arrays
    red_points = np.array(list(red_coords))
    yellow_points = np.array(list(yellow_coords))
    green_points = np.array(list(green_coords))

    return red_points, yellow_points, green_points

def create_map_layer(terrain_score_matrix, start_coords, red_points, yellow_points, green_points, folder, search_id=0):
    concave_hull_r = compute_concave_hull_from_points(red_points, BranchingConfig.HULL_ALPHA.value)
    concave_hull_y = compute_concave_hull_from_points(yellow_points, BranchingConfig.HULL_ALPHA.value)
    concave_hull_g = compute_concave_hull_from_points(green_points, BranchingConfig.HULL_ALPHA.value)
    
    red_75 = create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_r, color="red", crs="EPSG:4326", folder=folder, search_id=search_id)
    yellow_50 = create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_y, color="yellow", crs="EPSG:4326", folder=folder, search_id=search_id)
    green_25 = create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_g, color="green", crs="EPSG:4326", folder=folder, search_id=search_id)

    return (green_25, yellow_50, red_75)

    #create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_r, color="red", crs="EPSG:25833", folder=folder, search_id=search_id)