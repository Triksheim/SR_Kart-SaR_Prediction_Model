from SAR_model import SARModel
from utility import plot_array

"""
    
    This script is an example use for running the SARModel class locally with the given configuration file as paramters.   

"""


# SAR model configuration file
config = "config_file.txt"
# Create the SAR model
model = SARModel(config_file_path=config)

# Collect geo data from all sources(GeoNorge, OpenStreetMap), saves to file.
model.collect_all_geo_data()

# File paths for geo data
height_tiff_filepath = model.config.OUTPUT_FOLDER + f'id{model.config.SEARCH_ID}_{model.config.LAT}_{model.config.LNG}_height_composite.tif'
terrain_type_tiff_filepath = model.config.OUTPUT_FOLDER + f'id{model.config.SEARCH_ID}_{model.config.LAT}_{model.config.LNG}_terrain_composite.tif'
railway_filepath = model.config.ARRAY_FOLDER + f'id{model.config.SEARCH_ID}_osm_railway_data.npy'
building_filepath = model.config.ARRAY_FOLDER + f'id{model.config.SEARCH_ID}_osm_building_data.npy'
trail_filepath_osm = model.config.ARRAY_FOLDER + f'id{model.config.SEARCH_ID}_osm_trail_data.npy'
trail_filepath_gn = model.config.ARRAY_FOLDER + f'id{model.config.SEARCH_ID}_gn_trail_data.npy'

# Create height array from tiff file
height_array = model.create_height_array(height_tiff_filepath)
# Create terrain rgb array from tiff file
rgb_array = model.create_terrain_rgb_array(terrain_type_tiff_filepath)
# Encode terrain rgb array
encoded_terrain_matrix = model.encode_terrain_rgb_array(rgb_array)


# Add railways
terrain_matrix = model.add_matrix_data_to_terrain_matrix(
    railway_filepath, model.config.TERRAIN_TYPE["Jernbane"], encoded_terrain_matrix
)

# Add buildings
terrain_matrix = model.add_matrix_data_to_terrain_matrix(
    building_filepath, model.config.TERRAIN_TYPE["Bygning"], terrain_matrix
)

# Combine trails from both sources
combined_trails = model.functions.add_binary_matrix_data_to_matrix(
    trail_filepath_gn, model.config.TERRAIN_TYPE["Sti og vei"], trail_filepath_osm
)
# Add trails, get complete terrain matrix
terrain_matrix = model.add_matrix_data_to_terrain_matrix(
    combined_trails, model.config.TERRAIN_TYPE["Sti og vei"], terrain_matrix
)
plot_array(terrain_matrix, title="Terrain matrix")

# Calculate slope matrix
slope_matrix = model.calculate_slope_matrix()
# Nullify slopes on trails
modified_slope_matrix = model.functions.add_binary_matrix_data_to_matrix(
    combined_trails, model.config.TERRAIN_TYPE["Sti og vei"], slope_matrix
)
plot_array(modified_slope_matrix, title="Modified slope matrix")

# Combine terrain and slope matrices, get terrain score matrix
terrain_score_matrix = model.create_combined_terrain_score_matrix(terrain_matrix, modified_slope_matrix)
plot_array(terrain_score_matrix, title="Terrain score matrix")

# Start branching simulation for search areas
model.start_branching_simulation(terrain_score_matrix)

# Get results from simulation saved in model
r25_coords, r50_coords, r75_coords = model.convert_coord_sets_to_arrays()

# Compute sectors for search areas
sectors = model.compute_r25_r50_r75_sectors(r25_coords, r50_coords, r75_coords)
hull_r25, hull_r50, hull_r75 = sectors

model.plot_simulation_result(terrain_score_matrix, sectors, save=True)
model.plot_simulation_result(terrain_score_matrix, sectors)


# Create map polygon overlays for search areas
polygon_r25 = model.create_map_overlay(terrain_score_matrix, hull_r25, "green")
polygon_r50 = model.create_map_overlay(terrain_score_matrix, hull_r50, "yellow")
polygon_r75 = model.create_map_overlay(terrain_score_matrix, hull_r75, "red")

