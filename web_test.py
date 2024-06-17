"""
This script is designed to test the callable web model functions from the `web_model_functions` module.

Usage:
- Ensure that the `web_model_functions` module is available in the same directory as this script.
- Enter the desired search parameters (search_id, latitude, longitude, search radius, search type) in the designated variables.
- Run the script.

Note: 
This script assumes that the necessary directories do not exist and will create them.
Results will be saved in the `output` directory in the current working directory.

"""

from web_model_functions import collect_model_data, start_model, generate_search_sectors
import os


# Unique search ID (Can be any integer)
search_id = 100

# Create directories for output
base_dir = os.getcwd()
base_dir_out = f'{base_dir}/output/'
base_dir_id = f'{base_dir_out}ID{search_id}/'
os.makedirs(base_dir_id, exist_ok=False)
os.makedirs(f'{base_dir_id}array/', exist_ok=False)
os.makedirs(f'{base_dir_id}overlays/', exist_ok=False)
os.makedirs(f'{base_dir_id}logs/', exist_ok=False)
os.makedirs(f'{base_dir_id}overlays/sectors', exist_ok=False)

# Start GPS coordinates WGS84 (IPP)
lat = 68.44333              
lng = 17.52796

# Search distance/radius (25%, 50%, 75%) in meters
# Example values: d25 = 200, d50 = 600, d75 = 1500 (Must be accending order)
# Values should not exceed 6000 meters.
d25 = 200
d50 = 600
d75 = 1500

# Search type ('Hiker', 'Hunter', 'Children 4-6', 'Children 7-9', 'BerryPicker', 'Autism', 'Dementia', 'Suicidal')
# Types changes the threshold for terrain type and obstacle avoidance.
search_type = "Hiker"


collect_model_data(search_id, lat, lng, d25, d50, d75, base_dir_id)
start_model(search_id,lat, lng, d25,d50,d75, base_dir_id, search_type)
generate_search_sectors(search_id, lat, lng, base_dir_id)