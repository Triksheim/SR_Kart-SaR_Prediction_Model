from utility import *
from geo_services import *
from SAR_model_functions import *
from scipy.ndimage import maximum_filter
from matplotlib.patches import Circle
import math
import numpy as np
import matplotlib.pyplot as plt
import time


          


if __name__ == "__main__":
    start_time_main = time.perf_counter()
    ################### S A R - M O D E L - S E T T I N G S ###################
    
    output_folder = "output/"
    arrays_folder = f'{output_folder}array/'

    # Set to visulize all data in pyplots or set individual plots below
    plot = True

    # Set to collect data from GeoNorge and OSM
    collect = False
    plot_collect = False

    ## preprocessing functions start ## -----------------------------------------------------------

    run_all_pp = False   # Set to run all preprocessing functions or set individual functions below

    # Encode terrain type data to 0-1 values based on RGB values
    encode = run_all_pp or False
    plot_encoded = False
    terrain_type_encoding = {    # Encoded values for terrain type score
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
    terrain_rgb_values = {  # RGB values for terrain types (from GeoNorge data)
        "Skog": (158, 204, 115),
        "Åpen fastmark": (217, 217, 217),
        "Hav": (204, 254, 254),
        "Ferskvann": (145, 231, 255),
        "Myr": (181, 236, 252),
        "Bebygd": (252, 219, 214),
        "Sti og vei": (179, 120, 76),
        "Dyrket mark": (255, 247, 167)
    }
    

    # Add trail data to terrain data
    add_trails = run_all_pp or False

    # Reduce resolution of terrain and height data
    reduced_resolution = run_all_pp or False
    plot_reduced_resolution = False
    reduction_factor = 5                   # factor for reducing resolution (in both x and y direction)
    
    # Calculate steepness of the terrain
    calculate_steepness = run_all_pp or False
    plot_steepness = False
    normalization_cap = 10                  # cap steepness value for normalization

    # Combine terrain and steepness matrixes and perform filtering
    combine_matrix = run_all_pp or False
    plot_combined_matrix = False
    combination_method = "square"           # method for combining matrixes (mean, multiply, square)
    filter_matrix = run_all_pp or True   # apply max filter to combined matrix
    filter_size = 3                         # nxn filter size
    
    ## preprocessing functions end ## -----------------------------------------------------------


    # Straight line simulation
    straight_line_simulation = False
    plot_straight_line = False
    l_range_factor = 2                # factor for "max" travel distance

    # Branching simulation
    branching_simulation = True
    plot_branching = False
    scatter_endpoints = False           # scatter endpoints in plot
    branching_sim_iterations = 2      # number of iterations for each direction (iter * 8)
    b_range_factor = 2                  # factor for "max" travel distance
    hull_alpha = 15                     # "concavity" of the search area hull
    ring_25 = 20                        # percentage of max distance for 25% ring
    ring_50 = 50                        # percentage of max distance for 50% ring
    worse_terrain_threshold = 0.35       # threshold for branching when worse terrain (0-1)
    random_branching_chance = 10         # chance of random branching (n/100.000)

    # Create map overlay layer with CRS from branching simulation results
    create_map_layer = True if branching_simulation else False
    #create_map_layer = False
    


    # Map size, increase map_extention to get larger area by increasing number of squares in each direction
    square_radius = 500     # meter radius from center per map square, total 2*r x 2*r area
    map_extention = 2       # square extension from centre by 2*map_extention*square_radius in each direction


    # WGS84 coordinate for the center of the search area (last seen location of the missing person)
    start_lat = 68.266150
    start_lng = 14.537723
    
    #67.31458155986105, 14.477247863863925  keiservarden
    #68.44333623319275, 17.52796511156201   pumpvannet
    #68.26615067072053, 14.537723823348557  nedre svolværvatnet, fra prototype


    ################### S E T T I N G S - E N D ###################
    
    

   

    if collect:
        print("Data collection started...")
        start_time_collect = time.perf_counter()
        get_all_geo_data(start_lat, start_lng, square_radius, map_extention, output_folder)
        end_time_collect = time.perf_counter()
        print(f"Collecting data took {end_time_collect - start_time_collect} seconds")

        if plot or plot_collect:

            height_matrix = np.load(f'{arrays_folder}height_data.npy')
            plot_array(height_matrix, cmap='Greys', label="moh", title="Høydedata")

            try:
                trail_data = np.load(f'{arrays_folder}gn_trail_data.npy')
                #print(trail_data.shape)
                plot_array(trail_data, cmap='terrain', title="Stier GeoNorge")
            except FileNotFoundError:
                print("No GeoNorge trail data found")

            try:
                trail_data = np.load(f'{arrays_folder}osm_trail_data.npy')
                #print(trail_data.shape)
                plot_array(trail_data, cmap='terrain', title="Stier OSM")
            except FileNotFoundError:
                print("No OSM trail data found")




    if encode:
        terrain_rgb_file = "terrain_RGB_data.npy"
        
        start_time = time.perf_counter()
        terrain_encoding(terrain_type_encoding, terrain_rgb_values, terrain_rgb_file, arrays_folder)
        end_time = time.perf_counter()
        print(f"Encoding took {end_time - start_time} seconds")

        if plot or plot_encoded:
            terrain_type_matrix = np.load(f'{arrays_folder}terrain_data_encoded.npy')
            plot_array(terrain_type_matrix, cmap='terrain', label="Verdi(0-1)", title="Arealtype")
    

    if add_trails:
        terrain_type_matrix = np.load(f'{arrays_folder}terrain_data_encoded.npy')
        trail_file_gn = 'gn_trail_data.npy'
        trail_file_osm = 'osm_trail_data.npy'
        trail_files = (trail_file_gn, trail_file_osm)
        add_trails_data_to_terrain(terrain_type_matrix, trail_files, arrays_folder)

        if plot:
            plot_array(terrain_type_matrix, cmap='terrain', label="Verdi(0-1)", title="Arealtype med stier")

    if reduced_resolution:
        terrain_type_matrix = np.load(f'{arrays_folder}terrain_with_trails_matrix.npy')
        terrain_type_matrix = reduce_resolution(terrain_type_matrix, reduction_factor, method="mean")
        terrain_type_matrix[0][0] = 1  # assure max range of 1 for better visualization in plots

        height_matrix = np.load(f'{arrays_folder}height_data.npy')
        height_matrix = reduce_resolution(height_matrix, reduction_factor, method="mean")

        if plot_reduced_resolution:
            plot_array(terrain_type_matrix, cmap='terrain', label="Verdi(0-1)", title="Arealtype nedskalert")
            plot_array(height_matrix, cmap='Greys', label="moh", title="Høydedata nedskalert")


    if calculate_steepness:
        if not reduced_resolution:
            height_matrix = np.load(f'{arrays_folder}height_data.npy')
            
        steepness_matrix = calc_steepness(height_matrix)
        steepness_matrix[steepness_matrix > normalization_cap] = normalization_cap        # cap steepness values

        norm_steepness_matrix = steepness_matrix / normalization_cap                      # normalize values to 0-1
        inv_norm_steepness_matrix = 1 - norm_steepness_matrix                             # invert values
        inv_norm_steepness_matrix = inv_norm_steepness_matrix ** 2                        # square values for better separation at extremes
        
        #inv_norm_steepness_matrix[inv_norm_steepness_matrix < 0.1] = 0                    # set values below 0.1 to 0

        if plot or plot_steepness:
            plot_array(steepness_matrix, cmap='terrain', label="Gradering (endring)", title="Stigning med maksverdi 10")
            plot_array(inv_norm_steepness_matrix, cmap='terrain', label="Bratt  ->  Flatt", title="Normalisert stigning")
        
        # print steepnes matrix saved to
        print(f"Steepness matrix saved to: {arrays_folder}normalized_steepness_matrix.npy")
        np.save(f'{arrays_folder}normalized_steepness_matrix.npy', inv_norm_steepness_matrix)



    if combine_matrix:
        if not reduced_resolution:
            terrain_type_matrix = np.load(f'{arrays_folder}terrain_with_trails_matrix.npy')    # terrain type with trails
        if not calculate_steepness:
            inv_norm_steepness_matrix = np.load(f'{arrays_folder}normalized_steepness_matrix.npy')

        inv_norm_steepness_matrix[terrain_type_matrix == 1] = 1             # set steepness to 1 where there is a trail (ignore steepness)
        inv_norm_steepness_matrix[inv_norm_steepness_matrix <= 0.1] = 0     # steepness values below 0.1 are set to 0
        terrain_type_matrix[terrain_type_matrix <= 0.05] = 0                # set terrain values below 0.05 to 0

        combined_matrix = combine_matrixes(terrain_type_matrix, inv_norm_steepness_matrix, combination_method)

        #combined_matrix[combined_matrix < 0.05] = 0

        if filter_matrix:
            combined_matrix = maximum_filter(combined_matrix, size=filter_size)

        if plot or plot_combined_matrix:
            plot_array(combined_matrix, cmap='terrain', label="Vanskelig  ->  Lett", title="Kombinert matrise, terreng score")

        print(f"Combined terrain score matrix saved to: {arrays_folder}terrain_score_matrix.npy")
        np.save(f'{arrays_folder}terrain_score_matrix.npy', combined_matrix)



    if straight_line_simulation:
        terrain_score_matrix = np.load(f'{arrays_folder}terrain_score_matrix.npy')
        
        radius = terrain_score_matrix.shape[0] / 2
        center = (int(radius), int(radius))
        max_distance = radius * l_range_factor
        print(f"Max distance: {max_distance}")
       
        end_points_75 = []
        end_points_50 = []
        end_points_25 = []

        for n in range (1,4,1):
            remaining_energy = max_distance * n / 2

            for angle in range(360):
                radians = math.radians(angle)
                length = max(2*radius - center[0], 2*radius - center[1])
                end_x = int(center[0] + length * math.cos(radians))
                end_y = int(center[1] + length * math.sin(radians))

                end_point, _ = calc_travel_distance(terrain_score_matrix, remaining_energy, center, end_x, end_y)
                if n == 1:
                    end_points_75.append(end_point)
                elif n == 2:
                    end_points_50.append(end_point)
                elif n == 3:
                    end_points_25.append(end_point)

        if plot or plot_straight_line:
            plt.imshow(terrain_score_matrix, cmap='terrain', interpolation='nearest')
            circle = Circle((radius, radius), (radius-(radius/10)), color="blue", fill=False)
            plt.gca().add_patch(circle)

            for point in end_points_25:
                plt.plot([center[1], point[1]], [center[0], point[0]], color='red', linewidth=1)
            for point in end_points_50:
                plt.plot([center[1], point[1]], [center[0], point[0]], color='yellow', linewidth=1)
            for point in end_points_75:
                plt.plot([center[1], point[1]], [center[0], point[0]], color='green', linewidth=1)

            x_coords, y_coords = zip(*end_points_25)
            plt.scatter(y_coords, x_coords, color='red', s=5)
            x_coords, y_coords = zip(*end_points_50)
            plt.scatter(y_coords, x_coords, color='yellow', s=5)
            x_coords, y_coords = zip(*end_points_75)
            plt.scatter(y_coords, x_coords, color='green', s=5)
            
            plt.axis('equal')
            plt.title("Straight line result plot")
            plt.show()



    if branching_simulation:
        
        terrain_score_matrix = np.load(f'{arrays_folder}terrain_score_matrix.npy')
        
        radius = terrain_score_matrix.shape[0] / 2
        center = (int(radius), int(radius))
        max_distance = radius * b_range_factor
        max_energy = max_distance
        
        green_coords = set()
        yellow_coords = set()
        red_coords = set()
        branches = set()
        last_cutoff = set()
        sets = (green_coords, yellow_coords, red_coords, branches, last_cutoff)


        print("Branching simulation started...")
        start_time = time.perf_counter()
        for n in range(branching_sim_iterations):
            curr_dir = 1
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    move_direction = (i, j)
                    if move_direction != (0, 0):
                        print(f'Iteration {n+1}/{branching_sim_iterations}, Direction {curr_dir}/{8}')
                        curr_dir += 1
                        branches = set() # reset branches
                        sets = (green_coords, yellow_coords, red_coords, branches, last_cutoff)
                        branching_movement(terrain_score_matrix, (center[0], center[1]), move_direction, max_energy, max_energy, sets, ring_25, ring_50, worse_terrain_threshold, random_branching_chance)
                        
        end_time = time.perf_counter()


        print(f'Params: {branching_sim_iterations} iter, {b_range_factor} b_range, {worse_terrain_threshold} terrain_thrshld, {random_branching_chance} random_chance')
        print(f"Branching simulation took {end_time - start_time} seconds")
        print(f'Unique paths simulated: {len(red_coords)}')  # number of endpoints/paths
        debug_stats_print()

        # convert sets to np arrays
        red_points = np.array(list(red_coords))
        yellow_points = np.array(list(yellow_coords))
        green_points = np.array(list(green_coords))

        print(f'Calculating polygons based on simulation results...')
        # compute concave hulls
        concave_hull_r = compute_concave_hull_from_points(red_points, hull_alpha)
        concave_hull_y = compute_concave_hull_from_points(yellow_points, hull_alpha)
        concave_hull_g = compute_concave_hull_from_points(green_points, hull_alpha)


        if plot or plot_branching:
            plt.imshow(terrain_score_matrix, cmap='terrain', interpolation='nearest')
            plt.colorbar(label="Terreng: Vaskelig  ->  Lett")
            circle = Circle((radius, radius), (radius-(radius/10)), color="blue", fill=False)   # search area circle
            plt.gca().add_patch(circle)

            if concave_hull_r:
                x_r, y_r = get_polygon_coords_from_hull(concave_hull_r)
                plt.fill(x_r, y_r, edgecolor='r',linewidth=3, fill=False)
            if concave_hull_y:
                x_y, y_y = get_polygon_coords_from_hull(concave_hull_y)
                plt.fill(x_y, y_y, edgecolor='y',linewidth=3, fill=False)
            if concave_hull_g:
                x_g, y_g = get_polygon_coords_from_hull(concave_hull_g)
                plt.fill(x_g, y_g, edgecolor='g',linewidth=3, fill=False)

            # plot endpoints
            if scatter_endpoints:
                plt.scatter(red_points[:,0], red_points[:,1], color='red', s=5)
                plt.scatter(yellow_points[:,0], yellow_points[:,1], color='yellow', s=5)
                plt.scatter(green_points[:,0], green_points[:,1], color='green', s=5)
            
            plt.title("Branching result plot")
            plt.axis('equal')
            plt.show()
    


    if create_map_layer:
        start_coords = start_lat, start_lng  # matrix center
        map_diameter = ((map_extention * 2) + 1) * (square_radius * 2)  # Diameter of the map in meters
        distance_per_index =  map_diameter / terrain_score_matrix.shape[0]  # Meters per index in the matrix

        # create map overlays for red, yellow and green areas
        create_polygon_map_overlay(terrain_score_matrix, distance_per_index, start_coords, concave_hull_r, color="red", crs="EPSG:25833")
        create_polygon_map_overlay(terrain_score_matrix, distance_per_index, start_coords, concave_hull_y, color="yellow", crs="EPSG:25833")
        create_polygon_map_overlay(terrain_score_matrix, distance_per_index, start_coords, concave_hull_g, color="green", crs="EPSG:25833")

        create_polygon_map_overlay(terrain_score_matrix, distance_per_index, start_coords, concave_hull_r, color="red", crs="EPSG:4326")
        create_polygon_map_overlay(terrain_score_matrix, distance_per_index, start_coords, concave_hull_y, color="yellow", crs="EPSG:4326")
        create_polygon_map_overlay(terrain_score_matrix, distance_per_index, start_coords, concave_hull_g, color="green", crs="EPSG:4326")


    end_time_main = time.perf_counter()
    print(f"Main took {end_time_main - start_time_main} seconds")

        

    