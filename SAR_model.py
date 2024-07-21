from model_config import ModelConfig
from model_functions import ModelFunctions
from ext_data_collection import ModelGeoDataManager

import os
import numpy as np



class SARModel:
    """
        Class for the Search and Rescue model.
        Contains methods for collecting data, data processing, and running the SAR simulation.
        Subclasses ModelConfig, ModelFunctions, ModelGeoDataManager.
    """
    def __init__(self, config_file_path:str = None) -> None:
        self.config = ModelConfig(config_file_path)
        self.data_manager = ModelGeoDataManager(config=self.config)
        self.functions = ModelFunctions(model=self, config=self.config)
        self.setup_directories()
        self.clear_log_file()

        self.green_coords = set()
        self.yellow_coords = set()
        self.red_coords = set()
        self.cut_off_coords = set()
        
    def setup_directories(self) -> None:
        os.makedirs(self.config.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(self.config.ARRAY_FOLDER, exist_ok=True)
        os.makedirs(self.config.OVERLAY_FOLDER, exist_ok=True)
        os.makedirs(self.config.SECTOR_FOLDER, exist_ok=True)
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        
    def clear_log_file(self) -> None:
        with open(self.config.LOG_FILE, 'w') as file:
            file.write('')

    def print_config(self) -> None:
        print(self.config.__dict__)

    def collect_all_geo_data(self) -> None:
        self.data_manager.get_all_geo_data()

    def collect_terrain_type_data_geonorge(self) -> None:
        self.data_manager.get_terrain_type_map()

    def collect_elevation_data_geonorge(self) -> None:
        self.data_manager.get_height_map_geonorge()

    def collect_trail_data(self) -> None:
        self.collect_trail_data_gn()
        self.collect_trail_data_osm()

    def collect_trail_data_gn(self) -> None:
        self.data_manager.get_trail_map_geonorge()

    def collect_trail_data_osm(self) -> None:
        self.data_manager.get_trail_map_osm()

    def collect_building_data_osm(self) -> None:
        self.data_manager.get_buildings_osm()

    def collect_railway_data_osm(self) -> None:
        self.data_manager.get_railways_osm()

    def create_height_array(self, tiff_path=None) -> None:
        return self.functions.create_height_array(tiff_path)

    def create_terrain_rgb_array(self, tiff_path=None) -> np.ndarray:
        return self.functions.create_terrain_rgb_array(tiff_path)

    def encode_terrain_rgb_array(self, rgb_array=None) -> np.ndarray:
        return self.functions.encode_terrain_rgb_array(rgb_array)

    def add_matrix_data_to_terrain_matrix(self, matrix, terrain_value:float, terrain_matrix=None) -> np.ndarray:
        return self.functions.add_binary_matrix_data_to_matrix(matrix, terrain_value, terrain_matrix)

    def calculate_slope_matrix(self, height_matrix=None, norm_cap=None, square_factor=None) -> np.ndarray:
        slope_array = self.functions.calculate_slope_matrix(height_matrix, norm_cap, square_factor)
        return slope_array
    
    def create_combined_terrain_score_matrix(self, terrain_matrix:np.ndarray, slope_matrix:np.ndarray) -> np.ndarray:
        return self.functions.create_terrain_score_matrix(terrain_matrix, slope_matrix)
        
    def combine_matrix(self, matrix_a:np.ndarray, matrix_b:np.ndarray) -> np.ndarray:
        return self.functions.combine_matrix(matrix_a, matrix_b)
    
    def start_branching_simulation(self, terrain_score_matrix=None) -> None:
        self.functions.branching_simulation(terrain_score_matrix)

    def convert_coord_sets_to_arrays(self) -> tuple:
        green = self.functions.convert_coords_from_set_to_np_array(self.green_coords, 2)
        yellow = self.functions.convert_coords_from_set_to_np_array(self.yellow_coords, 5)
        red = self.functions.convert_coords_from_set_to_np_array(self.red_coords, 10)
        return green, yellow, red
    
    def compute_r25_r50_r75_sectors(self, r25_points, r50_points, r75_points ) -> tuple:
        sectors = self.functions.compute_r25_r50_r75_sectors(r25_points, r50_points, r75_points)
        return sectors
    
    def create_map_overlay(self, terrain_score_matrix, hull, color) -> None:
        self.functions.create_map_overlay(terrain_score_matrix, hull, color)

    def plot_simulation_result(self, terrain_score_matrix, sector_polygons, save=False) -> None:
        self.functions.plot_simulation_result(terrain_score_matrix, sector_polygons, save)

