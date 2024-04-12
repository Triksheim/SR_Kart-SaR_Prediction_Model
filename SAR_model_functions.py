import numpy as np
import math
import random

#############################################################
""" For debugging branching simulation """
cnt = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
def debug_stats_print():
    global cnt
    global cnt2
    global cnt3
    global cnt4
    print(f'{cnt} Steps')
    print(f'{cnt2} Branching, random')
    print(f'{cnt4} Branching, worse terrain')
    print(f'{cnt3} Direction change, obstacle')
#############################################################


def terrain_encoding(terrain_filename="terrain_RGB_data.npy", trail_gn_file="trail_data.npy", trail_osm_file="osm_trail_data.npy", folder="output/array/"):
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
        "Ukjent":       1,
        "Sti og vei":   1, 
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
             

    # add trails data to the terrain encoding. set 1 for trails
    try:
        trail_data_gn = np.load(f'{folder}{trail_gn_file}')
        terrain_type[trail_data_gn == 1] = 1
    except:
        print(f'No trail data found in {folder}{trail_gn_file}')

    try:
        trail_data_osm = np.load(f'{folder}{trail_osm_file}')
        terrain_type[trail_data_osm == 1] = 1
    except:
        print(f'No trail data found in {folder}{trail_osm_file}')

    np.save(f'{folder}terrain_data_encoded.npy', terrain_type)
    print(f'Encoded terrain data np array saved to {folder}terrain_data_encoded.npy')


def calc_steepness(height_matrix):
    # Calculate gradients along both axes
    grad_x, grad_y = np.gradient(height_matrix)
    # Calculate the steepness for each square
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

def branching_movement(matrix, current_idx, move_dir, initial_energy, current_energy, sets):
    global cnt, cnt2, cnt3, cnt4
    worse_terrain_threreshold = -0.25
    random_branching_chance = 100    # n/100.000


    green, yellow, red, branches = sets   # refrenced sets from main function
    x, y = current_idx
    if current_energy <= 0:
        red.add(current_idx) # red coords
        return

    # movement
    (new_x, new_y), movement_cost = movement(matrix, current_idx, move_dir)
    energy_left = current_energy - movement_cost

    # save green and yellow coords if energy dips threshold
    if current_energy > initial_energy*0.66:
        if energy_left < initial_energy*0.66:
            green.add((new_x, new_y))
    elif current_energy > initial_energy*0.33:
        if energy_left < initial_energy*0.33:
            yellow.add((new_x, new_y))


    if energy_left > 0:
        
        try:
            terrain_change = matrix[new_y, new_x] - matrix[y, x]
        except:
            terrain_change = 10  # prob out of bounds

        # Branching if terrain change is significant worse
        if terrain_change < worse_terrain_threreshold and (new_x, new_y) not in branches:
            cnt4 += 1

            sx, sy = move_dir   # current direction

            # continue in the same direction
            branching_movement(matrix, (new_x, new_y), move_dir, initial_energy, energy_left, sets)


            # Checks currents direction to avoid going in the opposite direction
            # Branches in 4 directions. diagonals in same direction and sideways.
            if abs(sx) == 1 and abs(sy) == 0:
                for i in range (0, 2, 1):
                    for j in range(-1, 2, 2):
                        new_dir = (i*sx, j)
                        branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                        branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets)

            elif abs(sx) == 0 and abs(sy) == 1:
                for i in range(-1, 2, 2):
                    for j in range(0, 2, 1):
                        new_dir = (i, j*sy)
                        branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                        branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets)

            elif abs(sx) == 1 and abs(sy) == 1:
                    new_dir = (0, sy)
                    branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                    branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets)

                    new_dir = (sx, 0)
                    branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                    branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets)

                    new_dir = (-sx, sy)
                    branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                    branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets)

                    new_dir = (sx, -sy)
                    branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                    branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets)



        # Random branching. Branches in all 8 directions
        elif random.randint(1,10000) <= random_branching_chance and (new_x, new_y) not in branches:
            cnt2 += 1
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    new_dir = (i, j)
                    if new_dir != (0, 0):
                        branches.add((new_x+new_dir[0], new_y+new_dir[1]))
                        branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets)
        

        # Stuck in the same position. Randomly choose a new direction
        elif (x, y) == (new_x, new_y) and (new_x, new_y) not in branches:
            new_dir = move_dir
            while new_dir == move_dir and new_dir != (0, 0):
                new_dir = (random.randint(-1, 1), random.randint(-1, 1))
            cnt3 += 1
            branching_movement(matrix, (new_x, new_y), new_dir, initial_energy, energy_left, sets)


        # Continue in the same direction 
        else:               
            branching_movement(matrix, (new_x, new_y), move_dir, initial_energy, energy_left, sets)
        
    else:
        # Energy depleted, stop recursion and save end point
        red.add((new_x, new_y)) # red coords
        


