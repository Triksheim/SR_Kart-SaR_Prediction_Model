from web_model_functions import *
import os


search_id = 250
# lat = 68.443336
# lng = 17.527965
lat = 68.44394111391593
lng = 17.52490997314453
d25 = 200
d50 = 500
d75 = 2500



base_dir = os.getcwd()
base_dir_out = f'{base_dir}/output/'
base_dir_id = f'{base_dir_out}ID{search_id}/'
os.makedirs(base_dir_id, exist_ok=False)
os.makedirs(f'{base_dir_id}array/', exist_ok=False)
os.makedirs(f'{base_dir_id}overlays/', exist_ok=False)
os.makedirs(f'{base_dir_id}logs/', exist_ok=False)






collect_model_data(search_id, lat, lng, d25, d50, d75, base_dir_id)
start_model(search_id,lat, lng, d25,d50,d75, base_dir_id)