from SAR_model_functions import  create_search_sectors_with_polygons
import numpy as np
import geopandas as gpd

# ONLY FOR TESTING SECTOR GENERATION

coords = 68.443336, 17.527965
#terrain_score_matrix = np.ones((1000, 1000))
terrain_score_matrix = np.load(f'output/array/id0_terrain_score_matrix.npy')
gdf = gpd.read_file(f'output/overlays/id0_red_68.443336_17.527965_EPSG4326.geojson')
gdf.to_crs('EPSG:25833', inplace=True)
hull_polygon = gdf.geometry[0]
print(f'{hull_polygon=}')

sector_max_size = 80_000 # m^2
redcution_factor = 5

# sectors = create_search_sectors(terrain_score_matrix, sector_max_size, redcution_factor)

# print(len(sectors))
# print(sectors[0].shape)

# print(sectors[0])

output_crs = 'EPSG:4326'

sector_polygons = create_search_sectors_with_polygons(
    terrain_score_matrix, coords, hull_polygon, sector_max_size, redcution_factor, output_crs)

print(f'Sectors created: {len(sector_polygons)}')
print(f'\nFirst sector: {sector_polygons[0]}')
      