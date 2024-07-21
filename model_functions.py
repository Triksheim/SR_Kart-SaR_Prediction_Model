from model_config import ModelConfig
#from SAR_model import SARModel
from utility import (
    exctract_data_from_tiff,
    reduce_resolution,
    write_to_log_file,
    downsample_rgb_image,
    matrix_value_padding,
    normalize_array,
    normalize_component,
    compute_concave_hull_from_points,
    create_polygon_map_overlay
    
)
import numpy as np
import time
import copy
import math
import random
import matplotlib.pyplot as plt


class ModelFunctions:
    """
    This class to handles the SAR models functions.
 
    """


    def __init__(self, model, config:ModelConfig) -> None:
        if not isinstance(config, ModelConfig):
            raise ValueError("The config parameter must be an instance of ModelConfig.")
        self.model = model
        self.config = config
        self.search_id = self.config.SEARCH_ID
        self.lat = self.config.LAT
        self.lng = self.config.LNG
        self.reduction_factor = self.config.REDUCTION_FACTOR
        self.output_folder = self.config.OUTPUT_FOLDER
        self.arrays_folder = self.config.ARRAY_FOLDER
        self.log_file = self.config.LOG_FILE


    def create_height_array(self, tiff_path=None, remove_last_row_col=True) -> np.ndarray:
            """
            Create a NumPy array from a TIFF file containing height data.

            Parameters:
            filepath (str, optional): The path to the TIFF file. If None, the default path is used.
            remove_last_row_col (bool, optional): Whether to remove the last row and column of the array.

            Returns:
            np.ndarray: The height array
            """
            write_to_log_file(f'{self.log_file}', f'Creating height array...')
            if tiff_path is None:
                tiff_path = f'{self.config.OUTPUT_FOLDER}id{self.search_id}_{self.lat}_{self.lng}_height_composite.tif'

            height_dataset = exctract_data_from_tiff(tiff_path)
            if remove_last_row_col:
                height_dataset = height_dataset[:-1,:-1]  # Remove last row and column
            height_dataset = reduce_resolution(height_dataset, factor=self.reduction_factor, method='mean')

            np.save(f'{self.arrays_folder}id{self.search_id}_height_matrix.npy', height_dataset)
            print(f'Height data np array saved to {self.arrays_folder}height_matrix.npy')
            write_to_log_file(f'{self.log_file}', f' done\n')
            return height_dataset

    
    def create_terrain_rgb_array(self, tiff_filepath=None) -> np.ndarray:
        """
        Create a RGB array from a terrain dataset stored in a TIFF file.

        Args:
            filepath (str, optional): The path to the TIFF file. If None, the default path is used.

        Returns:
            np.ndarray: The terrain RGB array.
        """
        write_to_log_file(f'{self.log_file}', f'Creating terrain RGB array...')

        if tiff_filepath is None:
            tiff_filepath = f'{self.output_folder}id{self.search_id}_{self.lat}_{self.lng}_terrain_composite.tif'

        terrain_dataset_R = exctract_data_from_tiff(tiff_path=tiff_filepath, band_n=1)
        terrain_dataset_G = exctract_data_from_tiff(tiff_path=tiff_filepath, band_n=2)
        terrain_dataset_B = exctract_data_from_tiff(tiff_path=tiff_filepath, band_n=3)
        terrain_dataset = np.array([terrain_dataset_R, terrain_dataset_G, terrain_dataset_B])

        # Downsample the terrain dataset
        terrain_dataset = downsample_rgb_image(terrain_dataset, factor=self.reduction_factor)

        np.save(f'{self.arrays_folder}id{self.search_id}_terrain_RGB_matrix.npy', terrain_dataset)
        print(f'Terrain RGB data np array saved to {self.arrays_folder}terrain_RGB_matrix.npy')
        write_to_log_file(f'{self.log_file}', f' done\n')
        return terrain_dataset


    def encode_terrain_rgb_array(self, rgb_array=None, terrain_type_encoding=None, terrain_rgb_values=None) -> np.ndarray:
        """
        Encode terrain types based on RGB values.

        Parameters:
        rgb_array (numpy.ndarray or filepath:str, optional): The RGB array to encode. If None, the default path is used.
        terrain_type_encoding (dict, optional): A dictionary mapping terrain names to their corresponding encodings.
        terrain_rgb_values (dict, optional): A dictionary mapping terrain names to their corresponding RGB values.

        Returns:
        np.ndarray: The encoded terrain matrix.
        """
        if not isinstance(rgb_array, np.ndarray):
            if rgb_array is None:
                rgb_array_filepath = f'{self.arrays_folder}id{self.search_id}_terrain_RGB_matrix.npy'
            else:
                rgb_array_filepath = rgb_array
            try:
                terrain_rgb_data_3d = np.load(f'{rgb_array_filepath}')
            except:
                print(f'No terrain data found in {rgb_array_filepath}')
                return

        else:
            terrain_rgb_data_3d = rgb_array

        write_to_log_file(f'{self.log_file}', f'Encoding terrain types...')
        print(f'Color analysis terrain encoding started...')
        print(f'{terrain_rgb_data_3d.shape[1]}x{terrain_rgb_data_3d.shape[2]} pixels to process.')

        if terrain_type_encoding is None:
            terrain_type_encoding = self.config.TERRAIN_TYPE
        if terrain_rgb_values is None:
            terrain_rgb_values = self.config.TERRAIN_RGB

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
        
        try:
            # Pad Y axis for swamps since they are horizontal lines seperated by white lines
            offset = [(1, 0), (-1, 0)]
            swamp_val = terrain_type_encoding["Myr"]
            matrix_value_padding(encoded_terrain, swamp_val, 1, offset)
        except:
            print("Swamp padding failed")
        
        encoded_terrain[0][0] = 1 # To assure that the min value is atleast 1 for better visual when plotting 

        # pad all roads and trails
        encoded_terrain = matrix_value_padding(encoded_terrain, terrain_type_encoding["Sti og vei"], 1)

        np.save(f'{self.arrays_folder}id{self.search_id}_terrain_type_matrix.npy', encoded_terrain)
        print(f'Encoded terrain data np array saved to {self.arrays_folder}id{self.search_id}_terrain_type_matrix.npy')
        write_to_log_file(f'{self.log_file}', f' done\n')
        return encoded_terrain


    def add_binary_matrix_data_to_matrix(self, binary_matrix, terrain_type_value, base_matrix=None) -> np.ndarray:
        """
        Add binary matrix data to the base matrix.
        
        Args:
            binary_matrix (numpy.ndarray or filepath): The binary matrix data to add to the base matrix.
            terrain_type_value (float): The terrain type value to assign to the truth matrix values.
            base_matrix (numpy.ndarray or filepath:str, optional): The matrix data to add data to, loads default path if None.
        """


        if not isinstance(binary_matrix, np.ndarray):
            try:
                binary_matrix = np.load(binary_matrix)
            except:
                print(f'No matrix data found in {binary_matrix}')
                return

        if not isinstance(base_matrix, np.ndarray):
            try:
                if base_matrix is None:
                    filepath = f'{self.arrays_folder}id{self.search_id}_terrain_type_matrix.npy'
                else:
                    filepath = base_matrix
            
                base_matrix = np.load(filepath)

            except:
                print(f'No terrain data found in {base_matrix}')
                return

        # Add matrix data to terrain data
        base_matrix[binary_matrix == 1] = terrain_type_value

        try:
            np.save(f'{self.arrays_folder}id{self.search_id}_terrain_type_matrix.npy', base_matrix)
            print(f'Matrix data added to terrain data and saved to {self.arrays_folder}id{self.search_id}_terrain_type_matrix.npy')
        except PermissionError: # if file is open
            while True:
                try:
                    np.save(f'{self.arrays_folder}id{self.search_id}_terrain_type_matrix.npy', base_matrix)
                    print(f'Matrix data added to terrain data and saved to {self.arrays_folder}id{self.search_id}_terrain_type_matrix.npy')
                    break
                except PermissionError:
                    time.sleep(1)

        return base_matrix


    def calculate_slope_matrix(self, height_matrix=None, norm_cap=None, square_factor=None) -> np.ndarray:
        """
        Create a slope matrix based on the given height matrix.

        Args:
            height_matrix (numpy.ndarray): The input height matrix.
            norm_cap (float): The maximum value for the slope matrix.
            square_factor (float): The factor to apply to the slope matrix.

        Returns:
            numpy.ndarray: The slope matrix.

        """
        if height_matrix is None:
            try:
                height_matrix = np.load(f'{self.arrays_folder}id{self.search_id}_height_matrix.npy')
            except:
                print(f'No height data found in {self.arrays_folder}id{self.search_id}_height_matrix.npy')
                return
        if norm_cap is None:
            norm_cap = self.config.NORMALIZE_CAP
        if square_factor is None:
            square_factor = self.config.SQUARE_FACTOR

        write_to_log_file(f'{self.log_file}', f'Calculating slope matrix...')


        # Gradients along both axes
        grad_x, grad_y = np.gradient(height_matrix)
        # Steepness for each square (pythagorean)
        slope_matrix = np.sqrt(grad_x**2 + grad_y**2)

        # Cap the slope matrix values
        slope_matrix[slope_matrix > norm_cap] = norm_cap

        # Normalize the slope matrix
        norm_slope_matrix = normalize_array(slope_matrix, norm_cap)

        # Inverse the slope matrix
        inv_slope_matrix = 1 - norm_slope_matrix
        inv_slope_matrix = inv_slope_matrix ** square_factor # more seperation between values

        np.save(f'{self.arrays_folder}id{self.search_id}_slope_matrix.npy', inv_slope_matrix)
        print(f'Slope data np array saved to {self.arrays_folder}id{self.search_id}_slope_matrix.npy')
        write_to_log_file(f'{self.log_file}', f' done\n')
        return inv_slope_matrix
    
    def create_terrain_score_matrix(self, terrain_matrix, slope_matrix) -> np.ndarray:
        """
        Create a terrain score matrix based on the terrain and slope matrices.

        Args:
            terrain_matrix (numpy.ndarray): The terrain matrix.
            slope_matrix (numpy.ndarray): The slope matrix.

        Returns:
            numpy.ndarray: The terrain score matrix.

        """
        write_to_log_file(f'{self.log_file}', f'Creating terrain score matrix...')
        terrain_score_matrix = self.combine_matrix(terrain_matrix, slope_matrix)
        np.save(f'{self.arrays_folder}id{self.search_id}_terrain_score_matrix.npy', terrain_score_matrix)
        print(f'Terrain score data np array saved to {self.arrays_folder}id{self.search_id}_terrain_score_matrix.npy')
        write_to_log_file(f'{self.log_file}', f' done\n')
        return terrain_score_matrix


    def combine_matrix(self, matrix1, matrix2, method="multiply") -> np.ndarray:
        if matrix1.shape != matrix2.shape:
            print("Matrixes are not the same size")
            return
        if method == "mean":
            return (matrix1 + matrix2) / 2
        elif method == "multiply":
            return matrix1 * matrix2
        

    def branching_simulation(self, terrain_score_matrix=None) -> None:
        """
        Simulates branching movement in the terrain score matrix based on given parameters.

        Args:
            terrain_score_matrix (numpy.ndarray or filepath, optional): Matrix representing the terrain score. If None, the default path is used.

        Returns:
            tuple: A tuple containing three numpy arrays representing the red points, yellow points, and green points.

        """
        if not isinstance(terrain_score_matrix, np.ndarray):
            try:
                terrain_score_matrix = np.load(terrain_score_matrix)
            except:
                try:
                    terrain_score_matrix = np.load(f'{self.arrays_folder}id{self.search_id}_terrain_score_matrix.npy')
                    print(f'Loaded base terrain score matrix from {self.arrays_folder}id{self.search_id}_terrain_score_matrix.npy')
                except:
                    print(f'No terrain score matrix')
                    return

        logfile = self.config.LOG_FILE
        worse_terrain_threshold = self.config.TERRAIN_CHANGE_THRESHOLD
        obstacle_threshold = self.config.OBSTACLE_THRESHOLD
        random_branching_chance = self.config.RANDOM_FACTOR
        
        radius = terrain_score_matrix.shape[0] / 2
        center = (int(radius), int(radius))

        ring_r25 = (self.config.R25 / self.config.REDUCTION_FACTOR) * self.config.RANGE_FACTOR
        ring_r50 = (self.config.R50 / self.config.REDUCTION_FACTOR) * self.config.RANGE_FACTOR
        ring_r75 = (self.config.R75 / self.config.REDUCTION_FACTOR) * self.config.RANGE_FACTOR

        max_distance = ring_r75
        
        # For logging progress
        start_time = time.perf_counter()
        write_to_log_file(f'{logfile}', f'\nBranching simulation started...')
        step_percentage = (100 / (self.config.ITERATIONS * 8))
        completion_percentage = 0
        with open(logfile, 'ab') as f:
            text = f'{completion_percentage}%'
            f.write(text.encode())

        time_limit = 300
        sim_start_time = time.perf_counter()

        # Calls branching_movement for each direction (8 directions in total) per iteration
        for n in range(self.config.ITERATIONS):
            curr_dir = 1
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    move_direction = (i, j)
                    if move_direction != (0, 0):
                        print(f'Iteration {n+1}/{self.config.ITERATIONS}, Direction {curr_dir}/{8}')
                        curr_dir += 1
                        if time.perf_counter() - sim_start_time > time_limit:
                            print(f'Time limit reached.')
                            break

                        green, yellow, red = self.branching_movement(
                            terrain_score_matrix, (center[0], center[1]), move_direction, max_distance, max_distance, ring_r25, ring_r50, worse_terrain_threshold, random_branching_chance, obstacle_threshold)
                        
                        # Save the coordinates for the current iteration to the model
                        self.model.green_coords.update(green)
                        self.model.yellow_coords.update(yellow)
                        self.model.red_coords.update(red)

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
            text = f'... 100%\n'
            log_step_back = len(text)
            f.seek(max(0, file_size - log_step_back), 0)  # Move pointer back
            f.write(text.encode())  # Write the final percentage
        end_time = time.perf_counter()
        write_to_log_file(f'{logfile}', f'Simulation done - Time: {end_time-start_time:0.2f}\n\n')

        

        
    

    def branching_movement(self, matrix, start_idx, move_dir, initial_move_resource, move_resource, ring_25, ring_50, terrain_change_threshold, random_branching_chance, obstacle_threshold) -> tuple:
        """
        Perform branching movement algorithm in a matrix.
        Saves the coordinates in different sets based on energy left.

        Args:
            matrix (numpy.ndarray): The matrix representing the terrain.
            start_idx (tuple): The starting index for the movement.
            move_dir (tuple): The initial movement direction.
            initial_move_resource (int): The initial movement resource.
            move_resource (int): The current movement resource.
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

        green = copy.copy(self.model.green_coords)
        yellow = copy.copy(self.model.yellow_coords)
        red = copy.copy(self.model.red_coords)
        last_cutoff = copy.copy(self.model.cut_off_coords)
        branch_log = set()

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
            (new_x, new_y), movement_cost = self.movement(matrix, current_idx, move_dir, obstacle_threshold)
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

        return green, yellow, red


    def movement(self, matrix, start_idx, move_dir, obstacle_threshold) -> tuple:
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
    

    def convert_coords_from_set_to_np_array(self, coord_set:set, reduction_factor=1) -> np.ndarray:
        return np.array(list(coord_set)[::reduction_factor])
    
    def create_map_layers(self, terrain_score_matrix, red_points, yellow_points, green_points):
        """
        Create map layers based on the terrain score matrix and point data.

        Args:
            terrain_score_matrix (numpy.ndarray): The terrain score matrix.
            red_points (list): List of red points.
            yellow_points (list): List of yellow points.
            green_points (list): List of green points.
            

        Returns:
            tuple: A tuple containing the map layers for 25%, 50%, and 75% confidence levels.
        """

        write_to_log_file(f'{self.log_file}', f'Creating map layers...')

        # Compute concave hulls from the red, yellow, and green points
        concave_hull_r = compute_concave_hull_from_points(red_points, self.config.HULL_ALPHA)
        concave_hull_y = compute_concave_hull_from_points(yellow_points, self.config.HULL_ALPHA)
        concave_hull_g = compute_concave_hull_from_points(green_points, self.config.HULL_ALPHA)
        
        start_coords = self.config.LAT, self.config.LNG
        search_id = self.config.SEARCH_ID
        folder = self.config.OVERLAY_FOLDER
        reduction_factor = self.config.REDUCTION_FACTOR

        # Create map overlays for the 25%, 50%, and 75% confidence levels with CRS EPSG:4326
        polygon_75 = create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_r, color="red", output_crs="EPSG:4326", folder=folder, reduction_factor=reduction_factor, search_id=search_id)
        polygon_50 = create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_y, color="yellow", output_crs="EPSG:4326", folder=folder, reduction_factor=reduction_factor, search_id=search_id)
        polygon_25 = create_polygon_map_overlay(terrain_score_matrix, start_coords, concave_hull_g, color="green", output_crs="EPSG:4326", folder=folder, reduction_factor=reduction_factor, search_id=search_id)

        write_to_log_file(f'{self.log_file}', f' done\n')
        return polygon_25, polygon_50, polygon_75
    
    def compute_r25_r50_r75_sectors(self, red_points, yellow_points, green_points, alpha=None):
        if alpha is None:
            alpha = self.config.HULL_ALPHA

       # Compute concave hulls from the green, yellow, and red points
        concave_hull_g = compute_concave_hull_from_points(green_points, alpha)
        concave_hull_y = compute_concave_hull_from_points(yellow_points, alpha)
        concave_hull_r = compute_concave_hull_from_points(red_points, alpha)

        return concave_hull_g, concave_hull_y, concave_hull_r
    
    def create_map_overlay(self, terrain_score_matrix, hull, color="black", output_crs="EPSG:4326"):
        write_to_log_file(f'{self.log_file}', f'Creating map overlay ({color})...')
        start_coords = self.config.LAT, self.config.LNG
        folder = self.config.OVERLAY_FOLDER
        search_id = self.config.SEARCH_ID
        reduction_factor = self.config.REDUCTION_FACTOR

        polygon = create_polygon_map_overlay(terrain_score_matrix, start_coords, hull, color=color, output_crs="EPSG:4326", folder=folder, reduction_factor=reduction_factor, search_id=search_id)
        write_to_log_file(f'{self.log_file}', f' done\n')
        return polygon
    

    def plot_simulation_result(self, terrain_score_matrix, sector_polygons, save=False):
        """
        Plot the simulation result with the terrain score matrix and the search sectors.

        Returns:
        None
        """
        plt.imshow(terrain_score_matrix, cmap='terrain', interpolation='nearest')
        matrix_radius = terrain_score_matrix.shape[0] / 2
        sector_radiuses = (self.config.R25, self.config.R50, self.config.R75)
        for sector_radius in sector_radiuses:
            circle = plt.Circle((matrix_radius, matrix_radius), (sector_radius/self.config.REDUCTION_FACTOR), color='black', fill=False)
            plt.gca().add_artist(circle)

        colors = ('g', 'y', 'r')
        for i, polygon in enumerate(sector_polygons):
            plt.plot(*polygon.exterior.xy, color=colors[i], linewidth=3)

        plt.title('Simulation Result')
        if save:
            plt.savefig(f'{self.config.LOG_DIR}id{self.config.SEARCH_ID}_simulation_result.png')
        else:
            plt.show()