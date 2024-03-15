from utility import *
from geo_services import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import uniform_filter




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
    #coverage_identifier = "nhm_dtm_topo_25833"
    #url = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"

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

    if not gdf_list:
        print("No trail data found")
        return

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
        "Dyrket mark": (255, 247, 167),
    }
    terrain_encoding = {
        "Ukjent":       1,
        "Sti og vei":   1, 
        "Åpen fastmark":0.8,
        "Bebygd":       0.8,
        "Dyrket mark":  0.7, 
        "Skog":         0.7,
        "Myr":          0.5,
        "Ferskvann":    0.1,
        "Hav":          0.05,   
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
                if terrain_type[i-1, j] == 0.3: # Myr
                    terrain_type[i, j] = terrain_encoding["Myr"]
                else:
                    terrain_type[i, j] = terrain_encoding[last_type]
                
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

    

    
              

def collect_data_test():
    start_lat = 68.443012
    start_lng = 17.527166
    rect_radius = 1000 # 1 km radius from center, total 2 km x 2 km area
    map_extention = 0   # extends from center point by 2*map_extention*rect_radius in each direction

    get_all_map_data(start_lat, start_lng, rect_radius, map_extention)

    



if __name__ == "__main__":
    collect = False
    encode = False
    heatmap_test = False

    if collect:
        collect_data_test()

    if encode:
        terrain_encoding()
    

    if heatmap_test:
        terrain_data = np.load("output/array/terrain_data_encoded.npy")
        terrain_data = reduce_resolution(terrain_data, factor=10, method="max")
        print(terrain_data.shape)
        #plot_array(terrain_data, cmap='terrain', label="Terreng")

        height_data = np.load("output/array/height_data.npy")
        height_data = height_data[:-1,:-1]
        height_data = reduce_resolution(height_data, factor=10, method="mean")

        steepness_map = calc_steepness(height_data)
        print(steepness_map.shape)
        #plot_array(steepness_map, cmap='terrain', label="Stigning (%)")

        normalized_steepness_map = steepness_map / np.max(steepness_map)
        normalized_steepness_map = 1 - normalized_steepness_map # Invert the steepness
        print(normalized_steepness_map.shape)
        #plot_array(normalized_steepness_map, cmap='terrain', label="Normalisert stigning")

        heatmap = combine_matrixes(terrain_data, normalized_steepness_map, method="square")
        print(heatmap.shape)
        #plot_array(heatmap, cmap='RdYlGn', label="Kombinert heatmap")
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()

        smoothed_heatmap = uniform_filter(heatmap, size=10)
        plt.imshow(smoothed_heatmap, cmap='hot', interpolation='nearest')
        plt.show()
    