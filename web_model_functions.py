try:
    from SAR_model_functions import (
        terrain_encoding,
        add_railway_data_to_terrain,
        add_building_data_to_terrain,
        add_trails_data_to_terrain,
        create_slope_matrix,
        combine_terrain_type_and_slope,
        branching_simulation,
        create_map_layer,
        calculate_map_extension,
        create_search_sectors_with_polygons
    )
    from utility import (
        plot_array,
        create_height_array,
        create_terrain_RGB_array,
    )
    from geo_services import get_all_geo_data
    from constants import ModelConfig
except ImportError:
    from .SAR_model_functions import (
        terrain_encoding,
        add_railway_data_to_terrain,
        add_building_data_to_terrain,
        add_trails_data_to_terrain,
        create_slope_matrix,
        combine_terrain_type_and_slope,
        branching_simulation,
        create_map_layer,
        calculate_map_extension,
        create_search_sectors_with_polygons
    )
    from .utility import (
        plot_array,
        create_height_array,
        create_terrain_RGB_array,
    )
    from .geo_services import get_all_geo_data
    from .constants import ModelConfig

import time
import numpy as np
import geopandas as gpd



# Run from webserver
def collect_model_data(search_id, lat, lng, d25, d50, d75, base_dir):

    # Set base dir
    config = ModelConfig(base_dir)

    logfile = f'{config.LOG_DIR}logfile.txt'
    with open(logfile, 'w') as f:
        f.write(f'SAR logfile: {search_id=}, {lat=}, {lng=}\n\n')

    get_model_data(search_id, lat, lng, d25, d50, d75, config)

# Run from webserver
def start_model(search_id, lat, lng, d25, d50, d75, base_dir):
    config = ModelConfig(base_dir, d25=d25, d50=d50, d75=d75)
    logfile = f'{config.LOG_DIR}logfile.txt'
    with open(logfile, 'a') as f:
        f.write(f'{config.config_str()}\n')
        


    layers = process_model_data(search_id, lat, lng, d25, d50, d75, config)
    return layers


def generate_search_sectors(search_id, lat, lng, base_dir):
    print(f'Generating search sectors...')
    config = ModelConfig(base_dir)
    terrain_score_matrix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_terrain_score_matrix.npy')
    coords = (lat, lng)

    gdf = gpd.read_file(f'{config.OVERLAY_FOLDER}id{search_id}_red_{lat}_{lng}_EPSG4326.geojson')
    hull_polygon = gdf.geometry[0]
    crs = 'EPSG:4326'


    sector_polygons = create_search_sectors_with_polygons(
        terrain_score_matrix, coords, hull_polygon, config.SECTOR_MAX_SIZE,
        config.REDUCTION_FACTOR, crs, config.SECTOR_FOLDER, search_id)
    
    print(f'Sectors created: {len(sector_polygons)}')
    return sector_polygons



def get_model_data(search_id, lat, lng, d25, d50, d75, config: ModelConfig):
    max_range = min(d75, 10000)
    map_extension = calculate_map_extension(max_range, config.SQUARE_RADIUS)
    print(f'{map_extension=}')


    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Collecting geo data...\n')
        f.write(f'Distance paramters: {d25=}, {d50=}, {d75=}, {map_extension=}\n')

    start_time = time.perf_counter()
    get_all_geo_data(search_id, lat, lng, config.SQUARE_RADIUS, map_extension, config.BASE_DIR, config.REDUCTION_FACTOR)
    end_time = time.perf_counter()
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Data collection done - Time: {end_time-start_time:.2f}\n\n')
    



def process_model_data(search_id, lat, lng, d25, d50, d75, config: ModelConfig):
    print(f'Processing model data for id: {search_id}')

    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Processing model data...\n')


    # Rasterize and encode terrain data
    start_time = time.perf_counter()
    # Create arrays from tiff files




    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Creating height array...')
    create_height_array(f'{config.BASE_DIR}id{search_id}_{lat}_{lng}_height_composite.tif', config.ARRAY_FOLDER,
                        config.REDUCTION_FACTOR, search_id)
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f' done\n')
    

    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Creating terrain RGB array...')
    create_terrain_RGB_array(f'{config.BASE_DIR}id{search_id}_{lat}_{lng}_terrain_composite.tif',
                            config.ARRAY_FOLDER,config.REDUCTION_FACTOR, search_id)
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f' done\n')
    

    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Encoding terrain types...')
    # Encode terrain type values
    terrain_rgb_file = f'{config.ARRAY_FOLDER}id{search_id}_terrain_RGB_matrix.npy'
    terrain_encoding(config.TERRAIN_TYPE, config.TERRAIN_RGB, terrain_rgb_file,
                      config.ARRAY_FOLDER, search_id)
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f' done\n')



    


    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Adding railways...')
    # Add railways
    #print("Adding railways to terrain data...")
    terrain_type_matrix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_terrain_type_matrix.npy')
    railway_files = [f'id{search_id}_osm_railway_data.npy']
    add_railway_data_to_terrain(terrain_type_matrix, railway_files, config.ARRAY_FOLDER,
                                 config.RAILWAY, search_id)
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f' done\n')


    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Adding buildings...')
    # Add buildings
    #rint("Adding buildings to terrain data...")
    terrain_type_matrix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_terrain_type_matrix.npy')
    building_files = [f'id{search_id}_osm_building_data.npy']
    add_building_data_to_terrain(terrain_type_matrix, building_files, config.ARRAY_FOLDER,
                                  config.BUILDING, search_id)
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f' done\n')


    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Adding trails...')
    # Add trails
    #print("Adding trails to terrain data...")
    terrain_type_matrix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_terrain_type_matrix.npy')
    trail_file_gn = f'id{search_id}_gn_trail_data.npy'
    trail_file_osm = f'id{search_id}_osm_trail_data.npy'
    trail_files = [trail_file_gn, trail_file_osm]
    add_trails_data_to_terrain(terrain_type_matrix, trail_files, config.ARRAY_FOLDER,
                                config.TRAIL, search_id)
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f' done\n')


    


    


    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Calculating slopes...')
    # Calculate slopes
    print("Calculating slopes...")
    height_matrix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_height_matrix.npy')
    slope_matrix = create_slope_matrix(height_matrix, config.NORMALIZE_CAP, config.SQUARE_FACTOR, config.ARRAY_FOLDER, search_id)
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f' done\n')



    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Creating terrain score matrix...')
    # Combine terrain and slope matrix
    print("Combining terrain and slope matrix...")
    terrain_type_matrix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_terrain_type_matrix.npy')
    terrain_score_marix = combine_terrain_type_and_slope(terrain_type_matrix, slope_matrix, config.COMBINATION_METHOD,
                                    config.ARRAY_FOLDER, search_id)
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f' done\n')
    end_time = time.perf_counter()
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Data processing done - Time: {end_time-start_time:.2f}\n\n')

    # Logging terrain score matrix png
    plot_array(array=terrain_score_marix, save=True, folder=config.LOG_DIR, title="Terrain score matrix")
    
        


    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Simulation started...\n')
    # Branching simulation
    print("Branching simulation started...")
    terrain_score_marix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_terrain_score_matrix.npy')
    start_time = time.perf_counter()
    red_points, yellow_points, green_points = branching_simulation(terrain_score_marix, search_id, d25, d50, d75, config)
    end_time = time.perf_counter()
    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Simulation done - Time: {end_time-start_time:.2f}\n\n')



    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'Creating map overlays...')
    # Calcuate polygons based on simulation
    print("Creating map overlays...")
    start_coords = (lat, lng)
    (layer_25,layer_50,layer_75) = create_map_layer(terrain_score_marix, start_coords, red_points, yellow_points, green_points, config.OVERLAY_FOLDER, search_id, config)

    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f' done\n\n')

    with open(f'{config.LOG_DIR}logfile.txt', 'a') as f:
        f.write(f'SAR model finished\n\n')


    return (layer_25,layer_50,layer_75)
    