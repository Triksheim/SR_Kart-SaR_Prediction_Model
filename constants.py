from enum import Enum

from pathlib import Path

#BASE_DIR = Path(__file__).resolve().parent.parent


class ModelConfig:
    
    def __init__(self, base_dir='output/', winter=False):
        self.BASE_DIR = base_dir
        self.ARRAY_FOLDER = f'{self.BASE_DIR}array/'
        self.OVERLAY_FOLDER = f'{self.BASE_DIR}overlays/'
        

        self.SQUARE_RADIUS = 500  # 1000m x 1000m square
        self.EXTRA_MAP_SIZE = 0.5  # 50% extra map size

        self.WINTER_MODE = winter

        self.REDUCTION_FACTOR = 5
        self.NORMALIZE_CAP = 10
        self.COMBINATION_METHORD = "square"
        self.FILTER_SIZE = 3

        self.ITERATIONS = 2
        self.RANGE_FACTOR = 2
        self.WORSE_TERRAIN = 0.3
        self.RANDOM_FACTOR = 10  # n/100.000
        self.HULL_ALPHA = 10


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





class PreProcessConfig(Enum):
    REDUCTION_FACTOR = 5
    NORMALIZE_CAP = 10
    COMBINATION_METHORD = "square"
    FILTER_SIZE = 3

class BranchingConfig(Enum):
    ITERATIONS = 2
    RANGE_FACTOR = 2
    WORSE_TERRAIN = 0.3
    RANDOM_FACTOR = 10  # n/100.000
    HULL_ALPHA = 10

class EncodingConfig(Enum):
    TRAIL = 1
    RAILWAY = 0.8
    BUILDING = 0
    TERRAIN_TYPE = {    # Encoded values for terrain type score
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
    TERRAIN_RGB = {  # RGB values for terrain types (from GeoNorge data)
        "Skog": (158, 204, 115),
        "Åpen fastmark": (217, 217, 217),
        "Hav": (204, 254, 254),
        "Ferskvann": (145, 231, 255),
        "Myr": (181, 236, 252),
        "Bebygd": (252, 219, 214),
        "Sti og vei": (179, 120, 76),
        "Dyrket mark": (255, 247, 167)
    }

