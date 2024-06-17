"""
This module contains the constants used in the model.
The `ModelConfig` class contains the configuration parameters for the model.
"""

class ModelConfig:
    
    def __init__(self, base_dir='output', winter=False, d25=None, d50=None, d75=None, search_type=None, sector_size=30_000):
        self.BASE_DIR = f'{base_dir}/'
        self.ARRAY_FOLDER = f'{self.BASE_DIR}array/'
        self.OVERLAY_FOLDER = f'{self.BASE_DIR}overlays/'
        self.LOG_DIR = f'{self.BASE_DIR}logs/'
        self.SECTOR_FOLDER = f'{self.OVERLAY_FOLDER}sectors/'
        self.LOG_FILE = f'{self.LOG_DIR}logfile.txt'
        
        self.SQUARE_RADIUS = 500  # 1000m x 1000m square
        self.EXTRA_MAP_SIZE = 0.5  # 50% extra map size

        self.SEARCH_TYPE = search_type if search_type else "Hiker"
        self.D25 = d25
        self.D50 = d50
        self.D75 = d75

        # Resolution
        self.REDUCTION_FACTOR = 5

        # Height
        self.NORMALIZE_CAP = 10
        self.SQUARE_FACTOR = 2

        # Matrix combination
        self.COMBINATION_METHOD = "multiply"

        # Siumlation
        self.ITERATIONS = 8
        self.RANGE_FACTOR = 1.25
        self.TERRAIN_CHANGE_THRESHOLD = 0.3
        self.OBSTACLE_THRESHOLD = self.get_params_for_category(self.SEARCH_TYPE)
        self.RANDOM_FACTOR = 2   
        self.HULL_ALPHA = 15

        # Search sectors
        self.SECTOR_MAX_SIZE = sector_size  # Default 30.000

        # Terrain type
        self.WINTER_MODE = winter
        self.TRAIL = 1
        self.RAILWAY = 0.8
        self.BUILDING = 0

        if self.WINTER_MODE:
            self.TERRAIN_TYPE = {    # Encoded values for terrain type score
                "Sti og vei":   1,
                "Ukjent":       0.8,
                "Åpen fastmark":0.8,
                "Bebygd":       0.8,
                "Dyrket mark":  0.8, 
                "Skog":         0.6,
                "Myr":          0.8,
                "Ferskvann":    0.8,
                "Hav":          0.01,   
            }
        else:
            self.TERRAIN_TYPE = {    # Encoded values for terrain type score
                "Sti og vei":   1,
                "Ukjent":       0.8,
                "Åpen fastmark":0.8,
                "Bebygd":       0.8,
                "Dyrket mark":  0.6, 
                "Skog":         0.6,
                "Myr":          0.3,
                "Ferskvann":    0.05,
                "Hav":          0.01,   
            }

        self.TERRAIN_RGB = {  # RGB values for terrain types (from GeoNorge data)
            "Skog": (158, 204, 115),
            "Åpen fastmark": (217, 217, 217),
            "Hav": (204, 254, 254),
            "Ferskvann": (145, 231, 255),
            "Myr": (181, 236, 252),
            "Bebygd": (252, 219, 214),
            "Sti og vei": (179, 120, 76),
            "Dyrket mark": (255, 247, 167)
        }

    def config_str(self):
        config_str = f"{self.SEARCH_TYPE=}, {self.D25=}, {self.D50=}, {self.D75=}, {self.RANGE_FACTOR=}\n"
        config_str += f"{self.REDUCTION_FACTOR=}, {self.NORMALIZE_CAP=}, {self.SQUARE_FACTOR=}, {self.COMBINATION_METHOD=}\n"
        config_str += f"{self.ITERATIONS=}, {self.TERRAIN_CHANGE_THRESHOLD=}, {self.OBSTACLE_THRESHOLD=}, {self.RANDOM_FACTOR=}, {self.HULL_ALPHA=}\n"
       
        return config_str


    def get_params_for_category(self, type):
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
        return terrain_tolerance[type]


