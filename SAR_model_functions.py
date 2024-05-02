import numpy as np
import math
import random
from utility import normalize_component

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


def terrain_encoding(terrain_type_encoding, terrain_rgb_values, terrain_filename="terrain_RGB_data.npy", folder="output/array/"):
    try:
        terrain_rgb_data_3d = np.load(f'{folder}{terrain_filename}')
    except:
        print(f'No terrain data found in {folder}{terrain_filename}')
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
    steps = 1
    set_neighbours(encoded_terrain, swamp_val, offset, steps)
    
    encoded_terrain[0][0] = 1 # To assure that the min value is atleast 1 for better visual when plotting 
    np.save(f'{folder}terrain_data_encoded.npy', encoded_terrain)
    print(f'Encoded terrain data np array saved to {folder}terrain_data_encoded.npy')

def terrain_encoding2(terrain_filename="terrain_RGB_data.npy", folder="output/array/"):
    """
    Encodes the terrain data based on RGB values and saves the encoded terrain data as a numpy array.

    Args:
        terrain_filename (str): The filename of the terrain RGB data numpy array. Default is "terrain_RGB_data.npy".
        trails_filename (str): The filename of the trail data numpy array. Default is "trail_data.npy".
        folder (str): The folder path where the numpy arrays are located. Default is "output/array/".

    Returns:
        None
    """
    
    try:
        terrain_data = np.load(f'{folder}{terrain_filename}')
    except:
        print(f'No terrain data found in {folder}{terrain_filename}')
        return
    
    # RGB values for terrain type
    terrain_rgb_values = {
        "Skog": (158, 204, 115),
        "Åpen fastmark": (217, 217, 217),
        "Hav": (204, 254, 254),
        "Ferskvann": (145, 231, 255),
        "Myr": (181, 236, 252),
        "Bebygd": (252, 219, 214),
        "Sti og vei": (179, 120, 76),
        "Dyrket mark": (255, 247, 167)
    }
    terrain_encoding = {
        "Sti og vei":   1,
        "Ukjent":       0.8,
        "Åpen fastmark":0.8,
        "Bebygd":       0.8,
        "Dyrket mark":  0.6, 
        "Skog":         0.6,
        "Myr":          0.3,
        "Ferskvann":    0.05,
        "Hav":          0.01,   
    }

    # create a new 2d array with the terrain type encoding based on rgb values
    terrain_type = np.zeros((terrain_data.shape[1], terrain_data.shape[2]), dtype=float)
    print(f'Color analysis terrain encoding started...')
    print(f'{terrain_data.shape[1]}x{terrain_data.shape[2]} pixels to process.')
    for i in range(terrain_data.shape[1]):
        for j in range(terrain_data.shape[2]):
            pixel_rgb = tuple(terrain_data[:, i, j])
            for terrain_name, rgb_value in terrain_rgb_values.items():
                if pixel_rgb == rgb_value:
                    # Use the reversed lookup to get the encoded integer
                    terrain_type[i, j] = terrain_encoding[terrain_name]
                    break

            if terrain_type[i, j] == 0:
                # check a if "Myr" is in a 3x3 area around the pixel
                if terrain_encoding["Myr"] in terrain_type[i-1:i+2, j-1:j+2]:
                    terrain_type[i, j] = terrain_encoding["Myr"]
                else:
                    terrain_type[i, j] = terrain_encoding["Åpen fastmark"]  # Unknown terrain type
                
        if i % (terrain_data.shape[1] / 100*5) == 0:
            if i != 0:
                print(f'{i/(terrain_data.shape[1]/100)}%')


    terrain_type[0][0] = 1 # To assure that the min value is atleast 1 for better visual when plotting 
    np.save(f'{folder}terrain_data_encoded.npy', terrain_type)
    print(f'Encoded terrain data np array saved to {folder}terrain_data_encoded.npy')



def add_trails_data_to_terrain(terrain_data, trail_files=[],folder="output/array/", trail_value=1):
        trails_added = False
        for trail_file in trail_files:
            try:
                trails = np.load(f'{folder}{trail_file}')
                terrain_data[trails == 1] = trail_value
                trails_added = True
            except:
                print(f'No trail data found in {folder}{trail_file}')
        
        if trails_added:
            neighbour_offset = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            steps = 8
            terrain_data = set_neighbours(terrain_data, trail_value, neighbour_offset, steps)
            np.save(f'{folder}terrain_with_trails_matrix.npy', terrain_data)
            print(f'Trails added to terrain data and saved to {folder}terrain_with_trails_matrix.npy')
        else:
            print(f'No trail data added to terrain data.')
            np.save(f'{folder}terrain_with_trails_matrix.npy', terrain_data)




def set_neighbours(matrix, value, neighbour_offset, steps=1):
    for _ in range(steps):
        rows, cols = np.where(matrix == value)
        
        for row, col in zip(rows, cols):
            for offset in neighbour_offset:
                new_row = row + offset[0]
                new_col = col + offset[1]
                try:
                    matrix[new_row, new_col] = value
                except:
                    pass
    return matrix
    

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



def combine_matrixes(terrain, steepness, method="mean"):
    if terrain.shape != steepness.shape:
        print("Matrixes are not the same size")
        return
    if method == "mean":
        return (terrain + steepness) / 2
    elif method == "multiply":
        return terrain * steepness
    elif method == "square":
        return (terrain * steepness) ** 2
    elif method == "cube":
        return (terrain * steepness) ** 3



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



def branching_movement(matrix, current_idx, move_dir, initial_energy, current_energy, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance):
    global cnt, cnt2, cnt3, cnt4, cnt_kill_1, cnt_kill_2, cnt_kill_3
    
    green, yellow, red, branches, last_cutoff = sets   # refrenced sets from main function
    x, y = current_idx
    if current_energy <= 0:
        red.add(current_idx) # red coords
        return

    # movement
    (new_x, new_y), movement_cost = movement(matrix, current_idx, move_dir)
    energy_left = current_energy - movement_cost

    # save green and yellow coords if energy dips threshold
    if current_energy > initial_energy*(100-ring_25)/100:
        if energy_left < initial_energy*(100-ring_25)/100:
            green.add((new_x, new_y))
    elif current_energy > initial_energy*(100-ring_50)/100:
        if energy_left < initial_energy*(100-ring_50)/100:
            if (new_x, new_y) in green:
                cnt_kill_1 += 1
                return
            else:
                yellow.add((new_x, new_y))


    elif current_energy > initial_energy*((100-ring_50)/2)/100:
        #print("Start of last cutoff")
        if energy_left < initial_energy*((100-ring_50)/2)/100:
            if (new_x, new_y) in green or (new_x, new_y) in yellow:
                #print(f'Killing branch: {new_x, new_y}')
                cnt_kill_2 += 1
                return
            else:
                #print(f'Adding to last cutoff: {new_x, new_y}')
                last_cutoff.add((new_x, new_y))


    
    # branches kill offs
    if current_energy < initial_energy*(100-((ring_50+ring_25)/2))/100:
        if (new_x, new_y) in green:
            cnt_kill_1 += 1
            return

    if current_energy < initial_energy*(100-ring_50)/100:
        if (new_x, new_y) in yellow:
            cnt_kill_2 += 1
            return

    if current_energy < initial_energy*((100-ring_50)/2)/100:
        if (new_x, new_y) in last_cutoff:
            cnt_kill_3 += 1
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
            branching_movement(matrix, (new_x, new_y), move_dir, initial_energy, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance)
            
            sx, sy = move_dir   # current direction
            # Branches in 4 directions. diagonals in same direction and sideways.
            new_directions = [
                (sy,-sx),   # right
                (-sy,sx),   # left
                (normalize_component(sx+sy),normalize_component(sy-sx)), # right diagonal
                (normalize_component(sx-sy),normalize_component(sy+sx))] # left diagonal

            for new_dir in new_directions:
                branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance)

            

        # Random branching. Branches in all 8 directions
        elif random.randint(1,10000) <= random_branching_chance and (new_x, new_y) not in branches:
            cnt2 += 1
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    new_dir = (i, j)
                    if new_dir != (0, 0):
                        branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                        branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance)
        
        # Change direction if obstacle, randomly choose right or left
        elif (x, y) == (new_x, new_y) and (new_x, new_y) not in branches:
            cnt3 += 1
            if random.randint(0,1) == 1:
                new_dir = (move_dir[1], -move_dir[0]) # right
            else:
                new_dir = (-move_dir[1], move_dir[0])  # left
            branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance)

        

        # Continue in the same direction 
        else:               
            branching_movement(matrix, (new_x, new_y), move_dir, initial_energy, energy_left, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance)
        
    else:
        # Energy depleted, stop recursion and save end point
        red.add((new_x, new_y)) # red coords
        


