from sarModel.modelFunctions.SAR_model_functions import *
from sarModel.modelFunctions.geo_services import *
from sarModel.modelFunctions.constants import *
import threading
import time
import os




# Run from webserver
def start_model(search_id, lat, lng, d25, d50, d75):
    get_model_data(search_id, (lat, lng), (d75, d50, d25))


def get_model_data(search_id=0, start_coordinates=(68.443336, 17.527965), ranges=(3200,1800,600)):
    lat, lng = start_coordinates
    max_range = ranges[0]
    map_extension = calculate_map_extension(max_range, ModelConfig.SQUARE_RADIUS.value, ModelConfig.EXTRA_MAP_SIZE.value)
    print(map_extension)

    # get geo data from API requests
    collect_args = (search_id, lat, lng, ModelConfig.SQUARE_RADIUS.value, map_extension, ModelConfig.OUTPUT_FOLDER.value)
    collect_thread = threading.Thread(target=get_all_geo_data, args=(collect_args))
    collect_thread.start()

    # Check if the data is available in db
    check_thread = threading.Thread(target=check_model_data, args=(search_id, lat, lng))
    check_thread.start()
    


def check_model_data(search_id, lat, lng):
    while True:
        try:
            # Check if the data is available for search_id
            time.sleep(10)
            
            if not os.path.exists(f'./{ModelConfig.OUTPUT_FOLDER.value}id{search_id}_{lat}_{lng}_terrain_composite.tif'):
                print(f'id{search_id}_{lat}_{lng}_terrain_composite.tif not found in {ModelConfig.OUTPUT_FOLDER.value}')
                continue
            if not os.path.exists(f'./{ModelConfig.OUTPUT_FOLDER.value}id{search_id}_{lat}_{lng}_height_composite.tif'):
                print(f'id{search_id}_{lat}_{lng}_height_composite.tif not found in {ModelConfig.OUTPUT_FOLDER.value}')
                continue
            if not os.path.exists(f'./{ModelConfig.ARRAY_FOLDER.value}id{search_id}_gn_trail_data.npy'):
                print(f'id{search_id}_gn_trail_data.npy not found in {ModelConfig.ARRAY_FOLDER.value}')
                continue
            if not os.path.exists(f'./{ModelConfig.ARRAY_FOLDER.value}id{search_id}_osm_trail_data.npy'):
                print(f'id{search_id}_osm_trail_data.npy not found in {ModelConfig.ARRAY_FOLDER.value}')
                continue
            if not os.path.exists(f'./{ModelConfig.ARRAY_FOLDER.value}id{search_id}_osm_building_data.npy'):
                print(f'id{search_id}_osm_building_data.npy not found in {ModelConfig.ARRAY_FOLDER.value}')
                continue
            if not os.path.exists(f'./{ModelConfig.ARRAY_FOLDER.value}id{search_id}_osm_railway_data.npy'):
                print(f'id{search_id}_osm_railway_data.npy not found in {ModelConfig.ARRAY_FOLDER.value}')
                continue

            # If all files are found, break the loop
            print(f'All files found for id: {search_id}')
            break

        except:
            print(f'Error checking model data for id: {search_id}')
            time.sleep(10)

    # Process the data
    process_model_data(search_id, lat, lng)



def process_model_data(search_id, lat, lng):
    print(f'Processing model data for id: {search_id}')

    # Rasterize and encode terrain data
    start_time = time.perf_counter()
    # Create arrays from tiff files
    create_height_array(f'./{ModelConfig.OUTPUT_FOLDER.value}id{search_id}_{lat}_{lng}_height_composite.tif',
                         ModelConfig.ARRAY_FOLDER.value, search_id)
    create_terrain_RGB_array(f'./{ModelConfig.OUTPUT_FOLDER.value}id{search_id}_{lat}_{lng}_terrain_composite.tif',
                              ModelConfig.ARRAY_FOLDER.value, search_id)
    # Encode terrain type values
    terrain_rgb_file = f'{ModelConfig.ARRAY_FOLDER.value}id{search_id}_terrain_RGB_matrix.npy'
    terrain_encoding(EncodingConfig.TERRAIN_TYPE.value, EncodingConfig.TERRAIN_RGB.value, terrain_rgb_file,
                      ModelConfig.ARRAY_FOLDER.value, search_id)
    end_time = time.perf_counter()
    print(f"Encoding took {end_time - start_time} seconds")

    # Add trails
    print("Adding trails to terrain data...")
    terrain_type_matrix = np.load(f'{ModelConfig.ARRAY_FOLDER.value}id{search_id}_terrain_data_encoded.npy')
    trail_file_gn = f'id{search_id}_gn_trail_data.npy'
    trail_file_osm = f'id{search_id}_osm_trail_data.npy'
    trail_files = [trail_file_gn, trail_file_osm]
    add_trails_data_to_terrain(terrain_type_matrix, trail_files, ModelConfig.ARRAY_FOLDER.value,
                                EncodingConfig.TRAIL.value, search_id)

    # Add buildings
    print("Adding buildings to terrain data...")
    building_files = [f'id{search_id}_osm_building_data.npy']
    add_building_data_to_terrain(terrain_type_matrix, building_files, ModelConfig.ARRAY_FOLDER.value,
                                  EncodingConfig.BUILDING.value, search_id)

    # Add railways
    print("Adding railways to terrain data...")
    railway_files = [f'id{search_id}_osm_railway_data.npy']
    add_railway_data_to_terrain(terrain_type_matrix, railway_files, ModelConfig.ARRAY_FOLDER.value,
                                 EncodingConfig.RAILWAY.value, search_id)

    # Reduce resolution
    print("Reducing resolution...")
    terrain_type_matrix = np.load(f'{ModelConfig.ARRAY_FOLDER.value}id{search_id}_terrain_type_matrix.npy')
    terrain_type_matrix = reduce_resolution(terrain_type_matrix, PreProcessConfig.REDUCTION_FACTOR.value)
    height_matrix = np.load(f'{ModelConfig.ARRAY_FOLDER.value}id{search_id}_height_matrix.npy')
    height_matrix = reduce_resolution(height_matrix, PreProcessConfig.REDUCTION_FACTOR.value)

    # Calculate slopes
    print("Calculating slopes...")
    slope_matrix = create_slope_matrix(height_matrix,PreProcessConfig.NORMALIZE_CAP.value, ModelConfig.ARRAY_FOLDER.value, search_id)

    # Combine terrain and slope matrix
    print("Combining terrain and slope matrix...")
    terrain_score_marix = combine_terrain_type_and_slope(terrain_type_matrix, slope_matrix, PreProcessConfig.COMBINATION_METHORD.value,
                                    PreProcessConfig.FILTER_SIZE.value, ModelConfig.ARRAY_FOLDER.value, search_id)
    
    # Branching simulation
    print("Branching simulation started...")
    red_points, yellow_points, green_points = branching_simulation(terrain_score_marix, search_id)

    # Calcuate polygons based on simulation
    print("Creating map overlays...")
    start_coords = (lat, lng)
    create_map_layer(terrain_score_marix, start_coords, red_points, yellow_points, green_points, ModelConfig.OVERLAY_FOLDER.value, search_id)

    
