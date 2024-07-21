"""
This module contains the config values used in the model.
The `ModelConfig` class contains the configuration parameters for the model.
"""

import ast
import os

class ModelConfig:
    """
    This class contains the configuration parameters for the model.
    The parameters are set to default values, which can be updated from a configuration file.
    
    """

    def __init__(self, config_file=None):
        # Default parameters
        model_params = {
            'search_id': 0,
            'winter': False,
            'lat': 68.44333,
            'lng': 17.52796,
            'r25': 600,
            'r50': 1200,
            'r75': 1800,
            'radius_limit': 10000,
            'search_category': 'Hiker',
            'sector_size': 30_000,
            'square_radius': 500,
            'extra_map_size': 0.5,
            'reduction_factor': 5,
            'normalize_cap': 10,
            'square_factor': 2,
            'combination_method': 'multiply',
            'iterations': 8,
            'range_factor': 1.25,
            'terrain_change_threshold': 0.3,
            'obstacle_threshold': 0.2,
            'random_factor': 2,
            'hull_alpha': 15,
            'base_dir': f'{os.getcwd()}\output',
        }

        # Update parameters from the configuration file
        if config_file:
            file_params = self.read_config_file(config_file)
            model_params.update(file_params)
            


        # Set all attributes
        self.SEARCH_ID = model_params['search_id']
        self.WINTER = model_params['winter']
        self.SEARCH_CATEGORY = model_params['search_category']
        self.LAT = model_params['lat']
        self.LNG = model_params['lng']
        self.R25 = model_params['r25']
        self.R50 = model_params['r50']
        self.R75 = model_params['r75']
        self.RADIUS_LIMIT = model_params['radius_limit']
        self.SECTOR_SIZE = model_params['sector_size']
        self.SQUARE_RADIUS = model_params['square_radius']
        self.EXTRA_MAP_SIZE = model_params['extra_map_size']
        self.REDUCTION_FACTOR = model_params['reduction_factor']
        self.NORMALIZE_CAP = model_params['normalize_cap']
        self.SQUARE_FACTOR = model_params['square_factor']
        self.COMBINATION_METHOD = model_params['combination_method']
        self.ITERATIONS = model_params['iterations']
        self.RANGE_FACTOR = model_params['range_factor']
        self.TERRAIN_CHANGE_THRESHOLD = model_params['terrain_change_threshold']
        self.RANDOM_FACTOR = model_params['random_factor']
        self.HULL_ALPHA = model_params['hull_alpha']
        self.OBSTACLE_THRESHOLD = self.get_obstacle_threshold_for_category(self.SEARCH_CATEGORY)

        # Path and directory setup
        self.BASE_DIR = model_params['base_dir'] + "\\"
        self.OUTPUT_FOLDER = f'{self.BASE_DIR}ID{self.SEARCH_ID}/'
        self.ARRAY_FOLDER = f'{self.OUTPUT_FOLDER}arrays/'
        self.OVERLAY_FOLDER = f'{self.OUTPUT_FOLDER}overlays/'
        self.SECTOR_FOLDER = f'{self.OVERLAY_FOLDER}sectors/'
        self.LOG_DIR = f'{self.OUTPUT_FOLDER}logs/'
        self.LOG_FILE = f'{self.LOG_DIR}logfile.txt'

        # Terrain types setup
        self.setup_terrain_types()

    def read_config_file(self, filename):
        try:
            with open(filename, 'r') as file:
                config_data = file.read()
                config_dict = ast.literal_eval(config_data)
                return config_dict
        except Exception as e:
            print(f"Failed to read or parse the configuration file: {e}")
            return {}

    def setup_terrain_types(self):
        # Adjust these types based on whether the winter mode is activated
        if self.WINTER:
            self.TERRAIN_TYPE = self.get_winter_terrain_type()
        else:
            self.TERRAIN_TYPE = self.get_default_terrain_type()

        self.TERRAIN_RGB = self.get_terrain_rgb_codes()

    def get_default_terrain_type(self):
        return {
            "Sti og vei": 1,
            "Ukjent": 0.8,
            "Åpen fastmark": 0.8,
            "Jernbane": 0.8,
            "Bebygd": 0.8,
            "Dyrket mark": 0.6, 
            "Skog": 0.6,
            "Myr": 0.3,
            "Ferskvann": 0.05,
            "Hav": 0.01,
            "Bygning": 0.00,
        }

    def get_winter_terrain_type(self):
        return {
            "Sti og vei": 1,
            "Ukjent": 0.8,
            "Åpen fastmark": 0.8,
            "Jernbane": 0.8,
            "Bebygd": 0.8,
            "Dyrket mark": 0.8, 
            "Skog": 0.6,
            "Myr": 0.8,
            "Ferskvann": 0.8,
            "Hav": 0.01,
            "Bygning": 0.00,
        }
    
    def get_terrain_rgb_codes(self):
        return {
            "Skog": (158, 204, 115),
            "Åpen fastmark": (217, 217, 217),
            "Hav": (204, 254, 254),
            "Ferskvann": (145, 231, 255),
            "Myr": (181, 236, 252),
            "Bebygd": (252, 219, 214),
            "Sti og vei": (179, 120, 76),
            "Dyrket mark": (255, 247, 167)
        }

    def get_obstacle_threshold_for_category(self, category):
        terrain_tolerance = { 
            # Terrain tolerance for different search types, lower equals more tolerant with bad terrain
            "Suicidal": 0.25,
            "Dementia": 0.30,
            "Autism": 0.10,
            "BerryPicker": 0.05,
            "Children 4-6": 0.10,
            "Children 7-9": 0.10,
            "Hiker": 0.20,
            "Hunter": 0.05,
        }
        return terrain_tolerance[category]




