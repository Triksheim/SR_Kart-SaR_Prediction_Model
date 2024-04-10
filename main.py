from utility import *
from geo_services import *
from SAR_model_functions import *

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import uniform_filter, generic_filter, maximum_filter
from matplotlib.patches import Circle
import math
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd






def get_height_map(center_of_map_squares, start_coords=("00.00","00.00"), rect_radius=1000, folder="output/"):
    """
    Retrieves height map data from a web service for a given set of map squares and saves a composite tiff to file.

    Args:
        center_of_map_squares (list): List of tuples representing the center coordinates of map squares.
        start_coords (tuple, optional): Starting coordinates. Defaults to ("00.00","00.00").
        rect_radius (int, optional): Radius of the rectangular area around each map square. Defaults to 1000.
        folder (str, optional): Output folder path. Defaults to "output/".

    Returns:
        None
    """
    
    # serivce 1 - DTM
    # coverage_identifier = "nhm_dtm_topo_25833"
    # url = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"

    # service 2 - DOM
    coverage_identifier = "nhm_dom_25833"
    url = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dom-nhm-25833"

    service_version = "1.1.2"
    response_format = "image/GeoTIFF" 

    tiff_data = []
    futures = []

    with ThreadPoolExecutor() as executor:
        print(f'Request to WCS: {url}')
        for coords in center_of_map_squares:
            time.sleep(5)
            bbox_coords = calculate_bbox(coords[0], coords[1], rect_radius)
            bbox = create_bbox_string(*bbox_coords)

            params = {
                "service": "WCS",
                "request": "GetCoverage",
                "version": service_version,
                "identifier": coverage_identifier,
                "format": response_format,
                "boundingbox": bbox,
            }

            # Submit task to the executor
            future = executor.submit(wcs_request, url, params)
            futures.append(future)

        # Wait for all futures to complete
        for n, future in enumerate(as_completed(futures)):
            response = future.result()
            if response:
                tiff_data.append(extract_tiff_from_multipart_response(response))
                print(f"{n+1}/{len(center_of_map_squares)}")

    # combine a composite tiff from all tiff data and save to file
    filename = f'{start_coords[0]}_{start_coords[1]}_height_composite.tif' 
    create_composite_tiff(tiff_data, f'{folder}{filename}')



def get_terrain_type_map(center_for_map_squares, start_coords=("00.00","00.00"), rect_radius=1000, folder="output/"):
    """
    Retrieves terrain type map from a WMS server based on the given parameters.

    Args:
        center_for_map_squares (list): List of coordinates representing the center points of map squares.
        start_coords (tuple, optional): Starting coordinates. Defaults to ("00.00","00.00").
        rect_radius (int, optional): Radius of the rectangular area to request from the WMS server. Defaults to 1000.
        folder (str, optional): Output folder path. Defaults to "output/".

    Returns:
        None
    """
    
    url = "https://wms.nibio.no/cgi-bin/ar5?language=nor"

    version = "1.3.0"
    layers = "Arealtype"    
    crs = "EPSG:25833"
    width = rect_radius*2
    height = rect_radius*2
    format = "image/tiff"

    images = []
    tiff_data = []
    futures = []

    with ThreadPoolExecutor() as executor:
        print(f'Request to WMS server: {url}')
        for n, coords in enumerate(center_for_map_squares):
            bbox_coords = calculate_bbox(coords[0], coords[1], rect_radius)
            bbox = f'{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}'

            params = {
            "service": "WMS",
            "request": "GetMap",
            "version": version,
            "layers":  layers,  
            "bbox": bbox,  # (minx, miny, maxx, maxy)
            "crs": crs,  
            "width": width,  
            "height": height,  
            "format": format  
            }
            
            future = executor.submit(wms_request, url, params)
            futures.append(future)
            

        for n, future in enumerate(as_completed(futures)):
            response = future.result()
            if response:
                image = Image.open(BytesIO(response.content))
                images.append(image)
                tiff_data.append(response.content)
                print(f"{n+1}/{len(center_for_map_squares)}")

        # create composite tiff/png and save to file    
        if format == "image/tiff":
            filename = f'{start_coords[0]}_{start_coords[1]}_terrain_composite.tif'
            filepath = f'{folder}{filename}' 
            create_composite_tiff(tiff_data, filepath)
        elif format == "image/png":
            filename = f'{start_coords[0]}_{start_coords[1]}_terrain_composite_image.png'
            filepath = f'{folder}{filename}' 
            create_composite_image(images, filepath)
    


def get_trail_map(bbox, folder="output/"):
    """
    Retrieves trail data within the specified bounding box and saves the trail map as an image and a NumPy array.

    Parameters:
    - bbox (tuple): The bounding box coordinates in the format (minx, miny, maxx, maxy).
    - folder (str): The folder path where the output files will be saved. Default is "output/".

    Returns:
    None
    """

    url = "https://wfs.geonorge.no/skwms1/wfs.turogfriluftsruter"
    feature_types = ["app:Fotrute", "app:AnnenRute", "app:Skiløype", "app:Sykkelrute"]

    bbox_str = create_bbox_string(*bbox)
    gdf_list = []

    print(f"Request to WFS server: https://wfs.geonorge.no/skwms1/wfs.turogfriluftsruter")
    for feature_type in feature_types:
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": feature_type,
            "bbox": bbox_str,
            "outputFormat": "application/gml+xml; version=3.2"
        }
        response = wfs_request(url, params)
        try:
            gdf = gpd.read_file(BytesIO(response.content))
            gdf_list.append(gdf)
        except:
            continue

    if len(gdf_list) == 0:
        print("No trail data found")
        return

    else:
        combined_df = pd.concat([*gdf_list], ignore_index=True)
        gdf = gpd.GeoDataFrame(combined_df, geometry="geometry")
        #print(gdf.crs)
        gdf = gdf.to_crs(epsg=25833)

        # save gdf plot to file as image
        gdf.plot()
        plt.savefig(f'{folder}trails.png')
        print(f'Trail plot saved to {folder}trails.png')

        minx, miny, maxx, maxy = bbox
        width = int(maxx - minx)
        height = int(maxy - miny)
        transform = from_origin(minx, maxy, 1, 1)  # 1x1 meter resolution

        # rasterization
        raster = rasterize(
            [(shape, 1) for shape in gdf.geometry],
            out_shape=(height, width),
            fill=0,
            transform=transform,
            all_touched=True
        )
        # raster is now a 2D NumPy array with 1s for trails and 0s elsewhere
        np.save(f'{folder}array/trail_data.npy', raster)
        print(f'Trail data np array saved to {folder}array/trail_data.npy')


def terrain_encoding(terrain_filename="terrain_RGB_data.npy", trails_filename="trail_data.npy", folder="output/array/"):
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
    last_type = "Ukjent"
    for i in range(terrain_data.shape[1]):
        for j in range(terrain_data.shape[2]):
            pixel_rgb = tuple(terrain_data[:, i, j])
            for terrain_name, rgb_value in terrain_rgb_values.items():
                if pixel_rgb == rgb_value:
                    # Use the reversed lookup to get the encoded integer
                    terrain_type[i, j] = terrain_encoding[terrain_name]
                    last_type = terrain_name
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
             

    # add trails data to the terrain data. set 1 for trails
    try:
        trail_data = np.load(f'{folder}{trails_filename}')
        terrain_type[trail_data == 1] = 1
    except:
        print(f'No trail data found in {folder}{trails_filename}')

    np.save(f'{folder}terrain_data_encoded.npy', terrain_type)
    print(f'Encoded terrain data np array saved to {folder}terrain_data_encoded.npy')


def create_height_array(filepath, folder="output/array/"):
    """
    Create a NumPy array from a TIFF file containing height data.

    Parameters:
    filepath (str): The path to the TIFF file.
    folder (str, optional): The folder to save the NumPy array. Default is "output/array/".

    Returns:
    None
    """
    height_dataset = exctract_data_from_tiff(tiff_path=filepath)
    np.save(f'{folder}height_data.npy', height_dataset)
    print(f'Height data np array saved to {folder}height_data.npy')


def create_terrain_RGB_array(filepath, folder="output/array/"):
    """
    Create a RGB array from a terrain dataset stored in a TIFF file.

    Args:
        filepath (str): The path to the TIFF file.
        folder (str, optional): The folder to save the RGB array. Defaults to "output/array/".

    Returns:
        None
    """
    terrain_dataset_R = exctract_data_from_tiff(tiff_path=filepath, band_n=1)
    terrain_dataset_G = exctract_data_from_tiff(tiff_path=filepath, band_n=2)
    terrain_dataset_B = exctract_data_from_tiff(tiff_path=filepath, band_n=3)
    terrain_dataset = np.array([terrain_dataset_R, terrain_dataset_G, terrain_dataset_B])
    np.save(f'{folder}terrain_RGB_data.npy', terrain_dataset)
    print(f'Terrain RGB data np array saved to {folder}terrain_RGB_data.npy')


def get_all_map_data(lat, lng, rect_radius=1000, map_extention=0, folder="output/"):
    """
    Retrieves and saves various map data based on the given latitude and longitude coordinates.

    Args:
        lat (float): The latitude coordinate.
        lng (float): The longitude coordinate.
        rect_radius (int, optional): The radius of each rectangular map square. Defaults to 1000.
        map_extention (int, optional): The number of map squares in each direction from the center. Defaults to 0.
        folder (str, optional): The folder path to save the map data. Defaults to "output/".
    """

    start_coords = (lat, lng)
    full_map_radius = 2*map_extention*rect_radius + rect_radius

    center_x, center_y = transform_coordinates_to_utm(lat, lng)
    min_x, min_y, max_x, max_y = calculate_bbox(center_x, center_y, full_map_radius)
    complete_bbox = (min_x, min_y, max_x, max_y)

    center_of_map_squares = []
    for y in range(-map_extention, map_extention+1, 1):
        for x in range(-map_extention, map_extention+1, 1):
            center_of_map_squares.append((center_x + (2*x*rect_radius), center_y + (2*y*rect_radius)))

    # get the terrain type map (saves tiff file)
    get_terrain_type_map(center_of_map_squares, start_coords, rect_radius)

    # get the height map    (saves tiff file)
    get_height_map(center_of_map_squares, start_coords, rect_radius)

    # get paths and trails map (saves numpy file)
    get_trail_map(complete_bbox)
    
    # convert height tiff to numpy array and save to file
    create_height_array(f'{folder}{start_coords[0]}_{start_coords[1]}_height_composite.tif')

    # convert terrain tiff to 3d numpy array with RGB values and save to file
    create_terrain_RGB_array(f'{folder}{start_coords[0]}_{start_coords[1]}_terrain_composite.tif')

    

    
              





    




if __name__ == "__main__":
    ####################################################################################################
    plot = True


    collect = True

    r = True # Set to run all preprocessing functions
    encode = r or False
    reduced_resolution = r or False
    calculate_steepness = r or False
    combine_matrix = r or False
    smoothed_matrix = r or False

    straight_line_test = True
    branching_line_test = True

    create_map_overlay = True
    
    ####################################################################################################
    #67.31458155986105, 14.477247863863925  keiservarden
    #68.44333623319275, 17.52796511156201   pumpvannet
    #68.26615067072053, 14.537723823348557  fra prototype

    start_lat = 68.26615
    start_lng = 14.53772

    square_radius = 500     # meter radius from center, total 2*r x 2*r area
    map_extention = 0       # extends from center point by 2*map_extention*square_radius in each direction

    if collect:
        get_all_map_data(start_lat, start_lng, square_radius, map_extention)

        if plot:
            height_data = np.load("output/array/height_data.npy")
            plot_array(height_data, cmap='terrain', label="moh", title="Høydedata")

            try:
                trail_data = np.load("output/array/trail_data.npy")
                plot_array(trail_data, cmap='terrain', title="Stier")
            except FileNotFoundError:
                print("No trail data found")


    if encode:
        filename1 = "terrain_RGB_data.npy"
        filename2 = "trail_data.npy"
        output = "output/array/"
        terrain_encoding(filename1, filename2, output)
        if plot:
            terrain_data = np.load("output/array/terrain_data_encoded.npy")
            plot_array(terrain_data, cmap='terrain', label="Verdi(0-1)", title="Arealtype")
    

    if reduced_resolution:
        factor = 5
        terrain_data = np.load("output/array/terrain_data_encoded.npy")
        terrain_data = reduce_resolution(terrain_data, factor, method="max")

        height_data = np.load("output/array/height_data.npy")
        height_data = height_data[:-1,:-1]
        height_data = reduce_resolution(height_data, factor, method="mean")


    if calculate_steepness:
        if not reduced_resolution:
            height_data = np.load("output/array/height_data.npy")
            height_data = height_data[:-1,:-1]

        steepness_map = calc_steepness(height_data)
        
        
        
        steepness_map[steepness_map > 10] = 10  # Cap steepness value at 10
        normalized_steepness_map = steepness_map / np.max(steepness_map)
        normalized_steepness_map = 1 - normalized_steepness_map # Invert the steepness
        normalized_steepness_map = normalized_steepness_map ** 2
        

        if plot:
            plot_array(steepness_map, cmap='terrain', label="Gradering", title="Normalisert Stigning")
            plot_array(normalized_steepness_map, cmap='terrain', label="Bratt  ->  Flatt", title="Invers normalisert stigning")
        
        np.save("output/array/normalized_steepness_map.npy", normalized_steepness_map)


    if combine_matrix:
        if not reduced_resolution:
            terrain_data = np.load("output/array/terrain_data_encoded.npy")
        if not calculate_steepness:
            normalized_steepness_map = np.load("output/array/normalized_steepness_map.npy")


        #combined_matrix = combine_matrixes(terrain_data, normalized_steepness_map, method="square")

        new_steepness_map = normalized_steepness_map
        new_steepness_map[terrain_data == 1] = 1

        new_steepness_map[new_steepness_map <= 0.5] = 0

        new_terrain_map = terrain_data
        new_terrain_map[new_terrain_map <= 0.05] = 0

        combined_matrix = combine_matrixes(new_terrain_map, new_steepness_map, method="square")

        if smoothed_matrix:
            filter_size = 3     # nxn filter size
            #combined_matrix = uniform_filter(combined_matrix, size=filter_size)
            combined_matrix = maximum_filter(combined_matrix, size=filter_size)

        if plot:
            plot_array(combined_matrix, cmap='terrain', label="Vanskelig  ->  Lett", title="Kombinert matrise, terreng score")
            

        np.save("output/array/combined_matrix.npy", combined_matrix)



    if straight_line_test:
        if not reduced_resolution:
            terrain_data = np.load("output/array/terrain_data_encoded.npy")
            height_data = np.load("output/array/height_data.npy")
            height_data = height_data[:-1,:-1]

        if not calculate_steepness:
            normalized_steepness_map = np.load("output/array/normalized_steepness_map.npy")

        if not combine_matrix:
            combined_matrix = np.load("output/array/combined_matrix.npy")
        
        radius = combined_matrix.shape[0] / 2
        center = (int(radius), int(radius))
        dist_range_factor = 2.5
        max_distance = radius * dist_range_factor
       
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

        if plot:
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
            plt.title("Straight line heatmap")
            plt.show()



    if branching_line_test:
        if not reduced_resolution:
            terrain_data = np.load("output/array/terrain_data_encoded.npy")
            height_data = np.load("output/array/height_data.npy")
            height_data = height_data[:-1,:-1]

        if not calculate_steepness:
            normalized_steepness_map = np.load("output/array/normalized_steepness_map.npy")

        if not combine_matrix:
            combined_matrix = np.load("output/array/combined_matrix.npy")
        
        radius = combined_matrix.shape[0] / 2
        center = (int(radius), int(radius))
        dist_range_factor = 2.5    # deterimines "max" travel distance
        max_distance = radius * dist_range_factor
        max_energy = max_distance
        alpha = 25  # "concavity" of the search area hull

        

        green_coords = set()
        yellow_coords = set()
        red_coords = set()
        branches = set()
        
        sets = (green_coords, yellow_coords, red_coords, branches)

        print("Branching simulation started...")



        # old function
        # for initial_angle in range(360):
        #    branching(combined_matrix, center[0], center[1], initial_angle, max_energy, max_energy, sets)



        iterations = 1
        directions = 8
        total = iterations * directions
        
        for n in range(iterations):
            curr_dir = 1
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    move_direction = (i, j)
                    if move_direction != (0, 0):
                        print(f'Iteration {n+1}/{iterations}, Direction {curr_dir}/{directions}')
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
        concave_hull_r = compute_concave_hull_from_points(red_points, alpha)
        concave_hull_y = compute_concave_hull_from_points(yellow_points, alpha)
        concave_hull_g = compute_concave_hull_from_points(green_points, alpha)


        if plot:
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
            #plt.scatter(red_points[:,0], red_points[:,1], color='red', s=5)
            #plt.scatter(yellow_points[:,0], yellow_points[:,1], color='yellow', s=25)
            #plt.scatter(green_points[:,0], green_points[:,1], color='green', s=50)

            
            plt.title("Branching result plot")
            plt.axis('equal')
            
            plt.show()
    


    if create_map_overlay:

        start_coords = start_lat, start_lng  # matrix center
        map_diameter = ((map_extention * 2) + 1) * (square_radius * 2)  # Diameter of the map in meters
        distance_per_index =  map_diameter / combined_matrix.shape[0]  # Meters per index in the matrix

        # create map overlays for red, yellow and green areas
        create_polygon_map_overlay(combined_matrix, distance_per_index, start_coords, concave_hull_r, color="red")
        create_polygon_map_overlay(combined_matrix, distance_per_index, start_coords, concave_hull_y, color="yellow")
        create_polygon_map_overlay(combined_matrix, distance_per_index, start_coords, concave_hull_g, color="green")





        

    