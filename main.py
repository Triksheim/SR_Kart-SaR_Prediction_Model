from utility import *
from geo_services import *
from SAR_model_functions import *
from scipy.ndimage import maximum_filter
from matplotlib.patches import Circle
import math
import numpy as np
import matplotlib.pyplot as plt


          


if __name__ == "__main__":
    ################### S A R - M O D E L - S E T T I N G S ###################
    
    output_folder = "output/"
    arrays_folder = f'{output_folder}array/'

    # Set to visulize all data in pyplots or set individual plots below
    plot = True

    # Set to collect data from GeoNorge and OSM
    collect = True
    plot_collect = False

    ## preprocessing functions start ##
    run_all_pp = True   # Set to run all preprocessing functions or set individual functions below

    # Encode terrain type data to 0-1 values based on RGB values
    encode = run_all_pp or False
    plot_encoded = False

    # Reduce resolution of terrain and height data
    reduced_resolution = run_all_pp or True
    reduction_factor = 5                    # factor for reducing resolution
    
    # Calculate steepness of the terrain
    calculate_steepness = run_all_pp or True
    plot_steepness = False
    normalization_cap = 10                  # cap steepness value for normalization

    # Combine terrain and steepness matrixes and perform filtering
    combine_matrix = run_all_pp or True
    plot_combined_matrix = False
    combination_method = "square"           # method for combining matrixes (mean, multiply, square)
    filter_matrix = run_all_pp or True     # apply max filter to combined matrix
    filter_size = 3                         # nxn filter size
    
    ## preprocessing functions end ##


    # Straight line simulation
    straight_line_simulation = True
    plot_straight_line = False
    l_range_factor = 2.5                # factor for "max" travel distance

    # Branching simulation
    branching_simulation = True
    plot_branching = False
    branching_sim_iterations = 16       # number of iterations for each direction (iter * 8)
    b_range_factor = 2                  # factor for "max" travel distance
    hull_alpha = 25                     # "concavity" of the search area hull
    scatter_endpoints = True            # scatter endpoints in plot

    # Create map overlay layer with CRS from branching simulation results
    create_map_layer = True if branching_simulation else False
    


    # Map size, increase map_extention to get larger area by increasing number of squares in each direction
    square_radius = 500     # meter radius from center per map square, total 2*r x 2*r area
    map_extention = 0       # square extension from centre by 2*map_extention*square_radius in each direction


    # WGS84 coordinate for the center of the search area (last seen location of the missing person)
    start_lat = 68.44333
    start_lng = 17.52796

    #67.31458155986105, 14.477247863863925  keiservarden
    #68.44333623319275, 17.52796511156201   pumpvannet
    #68.26615067072053, 14.537723823348557  nedre svolværvatnet, fra prototype


    ################### S E T T I N G S - E N D ###################
    
    

   

    if collect:
        get_all_geo_data(start_lat, start_lng, square_radius, map_extention, output_folder)

        if plot or plot_collect:

            height_data = np.load(f'{arrays_folder}height_data.npy')
            plot_array(height_data, cmap='terrain', label="moh", title="Høydedata")

            try:
                trail_data = np.load(f'{arrays_folder}gn_trail_data.npy')
                print(trail_data.shape)
                plot_array(trail_data, cmap='terrain', title="Stier GeoNorge")
            except FileNotFoundError:
                print("No GeoNorge trail data found")

            try:
                trail_data = np.load(f'{arrays_folder}osm_trail_data.npy')
                print(trail_data.shape)
                plot_array(trail_data, cmap='terrain', title="Stier OSM")
            except FileNotFoundError:
                print("No OSM trail data found")



    if encode:
        terrain_file = "terrain_RGB_data.npy"
        trail_gn_file = "gn_trail_data.npy"
        trail_osm_file = "osm_trail_data.npy"
        terrain_encoding(terrain_file, trail_gn_file, trail_osm_file, arrays_folder)

        if plot or plot_encoded:
            terrain_data = np.load(f'{arrays_folder}terrain_data_encoded.npy')
            plot_array(terrain_data, cmap='terrain', label="Verdi(0-1)", title="Arealtype")
    


    if reduced_resolution:
        terrain_data = np.load(f'{arrays_folder}terrain_data_encoded.npy')
        terrain_data = reduce_resolution(terrain_data, reduction_factor, method="max")

        height_data = np.load(f'{arrays_folder}height_data.npy')
        height_data = height_data[:-1,:-1]
        height_data = reduce_resolution(height_data, reduction_factor, method="mean")



    if calculate_steepness:
        if not reduced_resolution:
            height_data = np.load(f'{arrays_folder}height_data.npy')
            height_data = height_data[:-1,:-1]

        norm_steepness_map = calc_steepness(height_data)
        norm_steepness_map[norm_steepness_map > normalization_cap] = normalization_cap  # cap steepness values

        inv_norm_steepness_map = norm_steepness_map / np.max(norm_steepness_map)    # normalize values to 0-1
        inv_norm_steepness_map = 1 - inv_norm_steepness_map     # invert values
        inv_norm_steepness_map = inv_norm_steepness_map ** 2    # square values for better separation at extremes
        
        if plot or plot_steepness:
            plot_array(norm_steepness_map, cmap='terrain', label="Gradering", title="Normalisert Stigning")
            plot_array(inv_norm_steepness_map, cmap='terrain', label="Bratt  ->  Flatt", title="Invers normalisert stigning")
        
        np.save(f'{arrays_folder}normalized_steepness_map.npy', inv_norm_steepness_map)



    if combine_matrix:
        if not reduced_resolution:
            terrain_data = np.load(f'{arrays_folder}terrain_data_encoded.npy')
        if not calculate_steepness:
            inv_norm_steepness_map = np.load(f'{arrays_folder}normalized_steepness_map.npy')

        inv_norm_steepness_map[terrain_data == 1] = 1   # set steepness to 1 where there is a trail (ignore steepness)
        inv_norm_steepness_map[inv_norm_steepness_map <= 0.1] = 0   # steepness values below 0.2 are set to 0

        terrain_map = terrain_data
        terrain_map[terrain_map <= 0.05] = 0    # set terrain values below 0.05 to 0

        combined_matrix = combine_matrixes(terrain_map, inv_norm_steepness_map, combination_method)

        if filter_matrix:
            combined_matrix = maximum_filter(combined_matrix, size=filter_size)

        if plot or plot_combined_matrix:
            plot_array(combined_matrix, cmap='terrain', label="Vanskelig  ->  Lett", title="Kombinert matrise, terreng score")
            
        np.save(f'{arrays_folder}combined_matrix.npy', combined_matrix)



    if straight_line_simulation:
        if not reduced_resolution:
            terrain_data = np.load(f'{arrays_folder}terrain_data_encoded.npy')
            height_data = np.load(f'{arrays_folder}height_data.npy')
            height_data = height_data[:-1,:-1]

        if not calculate_steepness:
            inv_norm_steepness_map = np.load(f'{arrays_folder}normalized_steepness_map.npy')

        if not combine_matrix:
            combined_matrix = np.load(f'{arrays_folder}combined_matrix.npy')
        
        radius = combined_matrix.shape[0] / 2
        center = (int(radius), int(radius))
        b_range_factor = 2.5
        max_distance = radius * b_range_factor
       
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

                end_point, _ = calc_travel_distance(combined_matrix, remaining_energy, center, end_x, end_y)
                if n == 1:
                    end_points_75.append(end_point)
                elif n == 2:
                    end_points_50.append(end_point)
                elif n == 3:
                    end_points_25.append(end_point)

        if plot or plot_straight_line:
            plt.imshow(combined_matrix, cmap='terrain', interpolation='nearest')
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
        if not reduced_resolution:
            terrain_data = np.load(f'{arrays_folder}terrain_data_encoded.npy')
            height_data = np.load(f'{arrays_folder}height_data.npy')
            height_data = height_data[:-1,:-1]

        if not calculate_steepness:
            inv_norm_steepness_map = np.load(f'{arrays_folder}normalized_steepness_map.npy')

        if not combine_matrix:
            combined_matrix = np.load(f'{arrays_folder}combined_matrix.npy')
        
        radius = combined_matrix.shape[0] / 2
        center = (int(radius), int(radius))
        max_distance = radius * b_range_factor
        max_energy = max_distance
        
        green_coords = set()
        yellow_coords = set()
        red_coords = set()
        branches = set()
        sets = (green_coords, yellow_coords, red_coords, branches)

        print("Branching simulation started...")
        for n in range(branching_sim_iterations):
            curr_dir = 1
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    move_direction = (i, j)
                    if move_direction != (0, 0):
                        print(f'Iteration {n+1}/{branching_sim_iterations}, Direction {curr_dir}/{8}')
                        curr_dir += 1
                        branches = set() # reset branches
                        sets = (green_coords, yellow_coords, red_coords, branches)
                        branching_movement(combined_matrix, (center[0], center[1]), move_direction, max_energy, max_energy, sets)
                        

        print(f'Unique paths simulated: {len(red_coords)}')  # number of endpoints/paths
        debug_stats_print()

        # convert sets to np arrays
        red_points = np.array(list(red_coords))
        yellow_points = np.array(list(yellow_coords))
        green_points = np.array(list(green_coords))

        # compute concave hulls
        concave_hull_r = compute_concave_hull_from_points(red_points, hull_alpha)
        concave_hull_y = compute_concave_hull_from_points(yellow_points, hull_alpha)
        concave_hull_g = compute_concave_hull_from_points(green_points, hull_alpha)


        if plot or plot_branching:
            plt.imshow(combined_matrix, cmap='terrain', interpolation='nearest')
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
        distance_per_index =  map_diameter / combined_matrix.shape[0]  # Meters per index in the matrix

        # create map overlays for red, yellow and green areas
        create_polygon_map_overlay(combined_matrix, distance_per_index, start_coords, concave_hull_r, color="red")
        create_polygon_map_overlay(combined_matrix, distance_per_index, start_coords, concave_hull_y, color="yellow")
        create_polygon_map_overlay(combined_matrix, distance_per_index, start_coords, concave_hull_g, color="green")





        

    