from web_model_functions import *

search_id = 50
lat = 68.443336
lng = 17.527965
d25 = 300
d50 = 600
d75 = 1500

base_dir = 'output/'


collect_model_data(search_id, lat, lng, d25, d50, d75, base_dir)
start_model(search_id,lat, lng, d25,d50,d75, base_dir)