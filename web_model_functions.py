try:
    from sarModel.modelFunctions.SAR_model_functions import *
    from sarModel.modelFunctions.geo_services import *
    from sarModel.modelFunctions.constants import *
except:
    from SAR_model_functions import *
    from geo_services import *
    from constants import *


import time



# Run from webserver
def collect_model_data(search_id, lat, lng, d25, d50, d75, base_dir):

    # Set base dir
    config = ModelConfig(base_dir)

    get_model_data(search_id, lat, lng, d25, d50, d75, config)

# Run from webserver
def start_model(search_id, lat, lng, d25, d50, d75, base_dir):
    config = ModelConfig(base_dir)
    layers = process_model_data(search_id, lat, lng, d25, d50, d75, config)
    return layers


    # #get_model_data(search_id, lat, lng, d25, d50, d75)
    # check_model_finished(search_id, lat, lng)
    # print('Model finished')

    # green_overlay_name = f'{ModelConfig.OVERLAY_FOLDER.value}id{search_id}_green_{lat}_{lng}_EPSG4326'
    # yellow_overlay_name = f'{ModelConfig.OVERLAY_FOLDER.value}id{search_id}_yellow_{lat}_{lng}_EPSG4326'
    # red_overlay_name = f'{ModelConfig.OVERLAY_FOLDER.value}id{search_id}_red_{lat}_{lng}_EPSG4326'

    # green_json = gpd.read_file(f'{green_overlay_name}.geojson')
    # yellow_json = gpd.read_file(f'{yellow_overlay_name}.geojson')
    # red_json = gpd.read_file(f'{red_overlay_name}.geojson')
    
    # green_shp = gpd.read_file(f'{green_overlay_name}.shp')
    # yellow_shp = gpd.read_file(f'{yellow_overlay_name}.shp')
    # red_shp = gpd.read_file(f'{red_overlay_name}.shp')

    # green_polygon = green_shp.geometry.iloc[0]
    # yellow_polygon = yellow_shp.geometry.iloc[0]
    # red_polygon = red_shp.geometry.iloc[0]
    
    # # poly_gdf = gpd.GeoDataFrame({'geometry': [green_polygon, yellow_polygon, red_polygon],
    # #                     'color': ['green', 'yellow', 'red']}, crs='EPSG:4326')

    # # # Plotting
    # # fig, ax = plt.subplots()
    # # for color, group in poly_gdf.groupby('color'):
    # #     group.plot(ax=ax, color=color, edgecolor='black')

    # # plt.show()



    # return (green_polygon,yellow_polygon,red_polygon), (green_json,yellow_json,red_json)

    

    
    





def get_model_data(search_id, lat, lng, d25, d50, d75, config):
    max_range = d75
    map_extension = calculate_map_extension(max_range, config.SQUARE_RADIUS)
    print(f'{map_extension=}')

    output_folder = config.BASE_DIR

    get_all_geo_data(search_id, lat, lng, config.SQUARE_RADIUS, map_extension, output_folder)

    # # get geo data from API requests
    # collect_args = (search_id, lat, lng, ModelConfig.SQUARE_RADIUS.value, map_extension, ModelConfig.OUTPUT_FOLDER.value)
    # collect_thread = threading.Thread(target=get_all_geo_data, args=(collect_args))
    # collect_thread.start()

    # # Check if the data is available in db
    # check_thread = threading.Thread(target=check_model_data, args=(search_id, lat, lng))
    # check_thread.start()
    
# def check_model_finished(search_id, lat, lng):
#     while True:
#         time.sleep(10)
#         try:
#             if not os.path.exists(f'{ModelConfig.OVERLAY_FOLDER.value}id{search_id}_green_{lat}_{lng}_EPSG4326.geojson'):
#                 # print os path
#                 print(f'{ModelConfig.OVERLAY_FOLDER.value}id{search_id}_green_{lat}_{lng}_EPSG4326.geojson')
                
#                 #print(f'id{search_id}_green_{lat}_{lng}_EPSG4326.geojson not found in {ModelConfig.OVERLAY_FOLDER.value}')
#                 continue
#             if not os.path.exists(f'{ModelConfig.OVERLAY_FOLDER.value}id{search_id}_yellow_{lat}_{lng}_EPSG4326.geojson'):
#                 #print(f'id{search_id}_yellow_{lat}_{lng}_EPSG4326.geojson not found in {ModelConfig.OVERLAY_FOLDER.value}')
#                 continue
#             if not os.path.exists(f'{ModelConfig.OVERLAY_FOLDER.value}id{search_id}_red_{lat}_{lng}_EPSG4326.geojson'):
#                 #print(f'id{search_id}_red_{lat}_{lng}_EPSG4326.geojson not found in {ModelConfig.OVERLAY_FOLDER.value}')
#                 continue


#             # If all files are found, break the loop
#             print(f'Overlay files found for id: {search_id}')
#             break

#         except:
#             print(f'Error checking overlay files for id: {search_id}')
#             time.sleep(10)

# def check_model_data(search_id, lat, lng):
#     while True:
#         try:
#             # Check if the data is available for search_id
#             time.sleep(10)
            
#             if not os.path.exists(f'{ModelConfig.OUTPUT_FOLDER.value}id{search_id}_{lat}_{lng}_terrain_composite.tif'):
#                 print(f'id{search_id}_{lat}_{lng}_terrain_composite.tif not found in {ModelConfig.OUTPUT_FOLDER.value}')
#                 continue
#             if not os.path.exists(f'{ModelConfig.OUTPUT_FOLDER.value}id{search_id}_{lat}_{lng}_height_composite.tif'):
#                 print(f'id{search_id}_{lat}_{lng}_height_composite.tif not found in {ModelConfig.OUTPUT_FOLDER.value}')
#                 continue
#             if not os.path.exists(f'{ModelConfig.ARRAY_FOLDER.value}id{search_id}_gn_trail_data.npy'):
#                 print(f'id{search_id}_gn_trail_data.npy not found in {ModelConfig.ARRAY_FOLDER.value}')
#                 continue
#             if not os.path.exists(f'{ModelConfig.ARRAY_FOLDER.value}id{search_id}_osm_trail_data.npy'):
#                 print(f'id{search_id}_osm_trail_data.npy not found in {ModelConfig.ARRAY_FOLDER.value}')
#                 continue
#             if not os.path.exists(f'{ModelConfig.ARRAY_FOLDER.value}id{search_id}_osm_building_data.npy'):
#                 print(f'id{search_id}_osm_building_data.npy not found in {ModelConfig.ARRAY_FOLDER.value}')
#                 continue
#             if not os.path.exists(f'{ModelConfig.ARRAY_FOLDER.value}id{search_id}_osm_railway_data.npy'):
#                 print(f'id{search_id}_osm_railway_data.npy not found in {ModelConfig.ARRAY_FOLDER.value}')
#                 continue

#             # If all files are found, break the loop
#             print(f'All files found for id: {search_id}')
#             break

#         except:
#             print(f'Error checking model data for id: {search_id}')
#             time.sleep(10)

#     # Process the data
#     process_model_data(search_id, lat, lng)



def process_model_data(search_id, lat, lng, d25, d50, d75, config):

    print(f'Processing model data for id: {search_id}')

    # Rasterize and encode terrain data
    start_time = time.perf_counter()
    # Create arrays from tiff files
    create_height_array(f'{config.BASE_DIR}id{search_id}_{lat}_{lng}_height_composite.tif', config.ARRAY_FOLDER,
                         search_id)
    create_terrain_RGB_array(f'{config.BASE_DIR}id{search_id}_{lat}_{lng}_terrain_composite.tif',
                              config.ARRAY_FOLDER, search_id)
    # Encode terrain type values
    terrain_rgb_file = f'{config.ARRAY_FOLDER}id{search_id}_terrain_RGB_matrix.npy'
    terrain_encoding(config.TERRAIN_TYPE, config.TERRAIN_RGB, terrain_rgb_file,
                      config.ARRAY_FOLDER, search_id)
    end_time = time.perf_counter()
    print(f"Encoding took {end_time - start_time} seconds")

    # Add trails
    print("Adding trails to terrain data...")
    terrain_type_matrix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_terrain_data_encoded.npy')
    trail_file_gn = f'id{search_id}_gn_trail_data.npy'
    trail_file_osm = f'id{search_id}_osm_trail_data.npy'
    trail_files = [trail_file_gn, trail_file_osm]
    add_trails_data_to_terrain(terrain_type_matrix, trail_files, config.ARRAY_FOLDER,
                                config.TRAIL, search_id)

    # Add buildings
    print("Adding buildings to terrain data...")
    building_files = [f'id{search_id}_osm_building_data.npy']
    add_building_data_to_terrain(terrain_type_matrix, building_files, config.ARRAY_FOLDER,
                                  config.BUILDING, search_id)

    # Add railways
    print("Adding railways to terrain data...")
    railway_files = [f'id{search_id}_osm_railway_data.npy']
    add_railway_data_to_terrain(terrain_type_matrix, railway_files, config.ARRAY_FOLDER,
                                 config.RAILWAY, search_id)

    # Reduce resolution
    print("Reducing resolution...")
    terrain_type_matrix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_terrain_type_matrix.npy')
    terrain_type_matrix = reduce_resolution(terrain_type_matrix, config.REDUCTION_FACTOR)
    height_matrix = np.load(f'{config.ARRAY_FOLDER}id{search_id}_height_matrix.npy')
    height_matrix = reduce_resolution(height_matrix, config.REDUCTION_FACTOR)

    # Calculate slopes
    print("Calculating slopes...")
    slope_matrix = create_slope_matrix(height_matrix,config.NORMALIZE_CAP, config.ARRAY_FOLDER, search_id)

    # Combine terrain and slope matrix
    print("Combining terrain and slope matrix...")
    terrain_score_marix = combine_terrain_type_and_slope(terrain_type_matrix, slope_matrix, config.COMBINATION_METHORD,
                                    config.FILTER_SIZE, config.ARRAY_FOLDER, search_id)
    
    # Branching simulation
    print("Branching simulation started...")
    red_points, yellow_points, green_points = branching_simulation(terrain_score_marix, search_id, d25, d50, d75, config)

    # Calcuate polygons based on simulation
    print("Creating map overlays...")
    start_coords = (lat, lng)
    (layer_25,layer_50,layer_75) = create_map_layer(terrain_score_marix, start_coords, red_points, yellow_points, green_points, config.OVERLAY_FOLDER, search_id, config)

    return (layer_25,layer_50,layer_75)
    