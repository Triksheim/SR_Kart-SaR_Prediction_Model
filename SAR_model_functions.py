import numpy as np
import math
import random


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



def calc_steepness(height_matrix):
    # Calculate gradients along both axes
    grad_x, grad_y = np.gradient(height_matrix)
    # Calculate the steepness for each square
    steepness_matrix = np.sqrt(grad_x**2 + grad_y**2)
    return steepness_matrix




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



# old
def traverse(matrix, energy, start, target_direction, step_limit=1):
    x0, y0 = start
    radians = math.radians(target_direction)
    
    # Calculate end points based on the step limit and direction
    dx = math.cos(radians) * step_limit
    dy = math.sin(radians) * step_limit
    # Theoretical end point considering no obstacles
    x1, y1 = x0 + dx, y0 + dy

    # Absolute values needed for Bresenham's algorithm
    dx = abs(dx)
    dy = abs(dy)
    
    # Determine the direction of steps (-1 for left/up, 1 for right/down)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    x, y = x0, y0
    energy_used = 0
    steps_count = 0
    err = dx - dy   # Error variable for Bresenham's line algorithm

    # Start traversal
    while True:
        # Check bounds to ensure the current position is within matrix
        if 0 <= x < matrix.shape[1] and 0 <= y < matrix.shape[0]:
            # Remember previous position before potential move
            prev_x, prev_y = x, y

            # If energy is depleted, step back to last valid position
            if energy_used > energy:
                x -= sx
                y -= sy
                break
        else:
            # If out of bounds, step back to last valid position and deplete remaining energy
            x -= sx
            y -= sy
            return (int(x), int(y)), 0
            #break

            

        # Bresenham's algorithm for line drawing
        e2 = 2 * err
        if e2 >= -dy:
            err -= dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

        try:
            if matrix[int(y)][int(x)] <= 0:
                #print("Obstacle detected at", (x, y))
                return (int(x), int(y)), 0
        except:
            return (int(prev_x), int(prev_y)), 0
        
        
        # Calculate energy cost based on the type of move and matrix values
        if (prev_x != x) and (prev_y != y):
            # Diagonal move, use √2 times the straight cost from the matrix
            energy_cost_diagonal = math.sqrt(2) / matrix[int(y)][int(x)]
            energy_used += energy_cost_diagonal
        else:
            # Straight move, cost is inverse of matrix value
            energy_cost_straight = 1 / matrix[int(y)][int(x)]
            energy_used += energy_cost_straight
        
        steps_count += 1

        # If target is reached or steps exceed the limit, stop the traversal
        if x == int(x1) and y == int(y1) or steps_count > step_limit:
            break

        
        
    energy_left = max(energy - energy_used, 0)
    return (int(x), int(y)), energy_left





# old
def branching(matrix, x, y, angle, initial_energy, current_energy, sets):
    green, yellow, red = sets   # refrenced sets from main function
    if current_energy <= 0:
        red.add((x, y)) # red coords
        return

    # movement
    steps = 1   # steps to move 
    (new_x, new_y), new_energy = traverse(matrix, current_energy, (x, y), angle, steps)

    # save green and yellow coords if energy dips threshold
    if current_energy > initial_energy*0.66:
        if new_energy < initial_energy*0.66:
            green.add((new_x, new_y))
    elif current_energy > initial_energy*0.33:
        if new_energy < initial_energy*0.33:
            yellow.add((new_x, new_y))

    if new_energy > 0:
        # Branching conditions
        try:
            terrain_change = matrix[new_x, new_y] - matrix[x, y]
        except:
            terrain_change = 0
        if terrain_change < -0.5 and (new_x, new_y) not in red:
            for i in range(-2, 3, 1):
                branch_angle = angle + i * 45  # Calculate the new direction
                branch_angle %= 360  # Normalize angle
                branching(matrix, new_x, new_y, branch_angle, initial_energy, new_energy, sets)
        # Random branching
        elif random.randint(1,100) <= 0:    
            if (new_x, new_y) not in red and (new_x, new_y) not in yellow and (new_x, new_y) not in green:
                for i in range(-5, 6, 1):
                    branch_angle = angle + i * 10  # Calculate the new direction
                    branch_angle %= 360  # Normalize angle
                    branching(matrix, new_x, new_y, branch_angle, initial_energy, new_energy, sets)
        else:
            # Continue in the same direction
            branching(matrix, new_x, new_y, angle, initial_energy, new_energy, sets)
        
    else:
        # Energy depleted, stop recursion and save end point
        red.add((new_x, new_y)) # red coords


    
    

def movement(matrix, start_idx, move_dir):
    global cnt
    cnt += 1
    x0, y0 = start_idx
    sx, sy = move_dir

    x = x0 + sx
    y = y0 + sy

    try:
        # obstacle, step back
        if matrix[y, x] <= 0:
            return (x0, y0), 10
            
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
    random_branching_chance = 10    # n/100.000


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
        


