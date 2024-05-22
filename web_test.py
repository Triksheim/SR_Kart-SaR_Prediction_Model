from web_model_functions import *
import os


search_id = 30
lat = 68.45993167852778
lng = 17.481715679168705
# lat = 68.26615067072053
# lng = 14.537723823348557
d25 = 800
d50 = 1600
d75 = 3200

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