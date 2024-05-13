from web_model_functions import *
import os


search_id = 101
lat = 68.443336
lng = 17.527965
d25 = 500
d50 = 1000
d75 = 1500



base_dir = os.getcwd()
base_dir_out = f'{base_dir}/output/'
base_dir_id = f'{base_dir_out}ID{search_id}/'
os.makedirs(base_dir_id, exist_ok=False)
os.makedirs(f'{base_dir_id}array/', exist_ok=False)
os.makedirs(f'{base_dir_id}overlays/', exist_ok=False)






collect_model_data(search_id, lat, lng, d25, d50, d75, base_dir_id)
start_model(search_id,lat, lng, d25,d50,d75, base_dir_id)