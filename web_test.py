from web_model_functions import *
import os

# For testing the web model functions, run this script

search_id = 2

# Start coordinates (IPP)
lat = 68.44333              
lng = 17.52796

# Search radius (25%, 50%, 75%) (meter)
d25 = 200
d50 = 600
d75 = 1500

search_type = "Hiker"

base_dir = os.getcwd()
base_dir_out = f'{base_dir}/output/'
base_dir_id = f'{base_dir_out}ID{search_id}/'
os.makedirs(base_dir_id, exist_ok=False)
os.makedirs(f'{base_dir_id}array/', exist_ok=False)
os.makedirs(f'{base_dir_id}overlays/', exist_ok=False)
os.makedirs(f'{base_dir_id}logs/', exist_ok=False)
os.makedirs(f'{base_dir_id}overlays/sectors', exist_ok=False)


collect_model_data(search_id, lat, lng, d25, d50, d75, base_dir_id)
start_model(search_id,lat, lng, d25,d50,d75, base_dir_id, search_type)

generate_search_sectors(search_id, lat, lng, base_dir_id)