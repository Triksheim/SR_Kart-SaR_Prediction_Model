from utility import *
import requests
from xml.etree import ElementTree as ET
from io import BytesIO
import time
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from rasterio.features import rasterize
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_all_geo_data(lat, lng, square_radius=1000, map_extention=0, folder="output/"):
    """
    Retrieves and saves various mgeo data based on the given latitude and longitude coordinates.

    Args:
        lat (float): The latitude coordinate.
        lng (float): The longitude coordinate.
        square_radius (int, optional): The radius of each rectangular map square. Defaults to 1000(m).
        map_extention (int, optional): The number of map squares in each direction from the center. Defaults to 0.
        folder (str, optional): The folder path to save the map data. Defaults to "output/".
    """

    start_point = (lat, lng)
    full_map_radius = 2*map_extention*square_radius + square_radius

    center_x, center_y = transform_coordinates_to_utm(lat, lng)
    min_x, min_y, max_x, max_y = calculate_bbox_utm(center_x, center_y, full_map_radius)
    utm33_bbox = (min_x, min_y, max_x, max_y)

    map_squares_center_point = []
    for y in range(-map_extention, map_extention+1, 1):
        for x in range(-map_extention, map_extention+1, 1):
            map_squares_center_point.append((center_x + (2*x*square_radius), center_y + (2*y*square_radius)))

     

    # get the terrain type map (saves tiff file)
    get_terrain_type_map(map_squares_center_point, start_point, square_radius, folder)

    # get the height map    (saves tiff file)
    get_height_map_geonorge(map_squares_center_point, start_point, square_radius, folder)

    # get paths and trails map (saves numpy file)
    get_trail_map_geonorge(utm33_bbox, folder)
    get_trail_map_osm(utm33_bbox, folder)
    
    # convert height tiff to numpy array and save to file
    create_height_array(f'{folder}{start_point[0]}_{start_point[1]}_height_composite.tif')

    # convert terrain tiff to 3d numpy array with RGB values and save to file
    create_terrain_RGB_array(f'{folder}{start_point[0]}_{start_point[1]}_terrain_composite.tif')



def get_height_map_geonorge(center_of_map_squares, start_coords, square_radius=1000, folder="output/"):
    """
    Retrieves height map data from a web service for a given set of map squares and saves a composite tiff to file.

    Args:
        center_of_map_squares (list): List of tuples representing the center coordinates of map squares.
        start_coords (tuple, optional): Starting coordinates.
        square_radius (int, optional): Radius of the area around each map square. Defaults to 1000.
        folder (str, optional): Output folder path. Defaults to "output/".

    Returns:
        None
    """
    
    # serivce 1 - DTM
    # coverage_identifier = "nhm_dtm_topo_25833"
    # url = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"

    # service 2 - DOM
    coverage_identifier = "nhm_dom_25833"
    url = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dom-nhm-25833"

    service_version = "1.1.2"
    response_format = "image/GeoTIFF" 

    tiff_data = []
    futures = []

    with ThreadPoolExecutor() as executor:
        print(f'Request to WCS: {url}')
        for coords in center_of_map_squares:
            time.sleep(5)
            bbox_coords = calculate_bbox_utm(coords[0], coords[1], square_radius)
            bbox = create_bbox_string(*bbox_coords)

            params = {
                "service": "WCS",
                "request": "GetCoverage",
                "version": service_version,
                "identifier": coverage_identifier,
                "format": response_format,
                "boundingbox": bbox,
            }

            # Submit task to the executor
            future = executor.submit(wcs_request, url, params)
            futures.append(future)

        # Wait for all futures to complete
        for n, future in enumerate(as_completed(futures)):
            response = future.result()
            if response:
                tiff_data.append(extract_tiff_from_multipart_response(response))
                print(f"{n+1}/{len(center_of_map_squares)}")

    # combine a composite tiff from all tiff data and save to file
    filename = f'{start_coords[0]}_{start_coords[1]}_height_composite.tif' 
    create_composite_tiff(tiff_data, f'{folder}{filename}')



def wcs_request(url, params, retry_count=10):
    """
    Make a request to a Web Coverage Service (WCS) and return the response.
        params = {
            "service": "WCS",
            "request": "GetCoverage",
            "version": service_version,
            "identifier": coverage_identifier,
            "format": response_format,
            "boundingbox": bbox
        }
        url = "source url"
    
    Example use @GeoNorge:
        params = {
            "service": "WCS",
            "request": "GetCoverage",
            "version": "1.1.2",
            "identifier": "nhm_dtm_topo_25833",  
            "format": "image/GeoTIFF",
            "boundingbox": "598524.15322,7592701.98006,600524.15322,7594701.98006,urn:ogc:def:crs:EPSG::25833"
        }
        url = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"
    """

    while retry_count > 0:
        # Make the request
        response = requests.get(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')
            if 'multipart' in content_type: # Check if the response is multipart xml
                if len(response.content) > 100000:
                    return response
                print(f'Error:Insufficient data. Count: {len(response.content)} bytes')
                
            else:
                print('Response unsuccessful. Content-Type is not multipart.')  # Uexpected response 
        else:
            print(f"Request failed with status code: {response.status_code}")

        retry_count -= 1
        print(f"Retrying request. Attempts left: {retry_count}")
        time.sleep(2)

    raise Exception("Request failed. No attempts left.")



def get_trail_map_geonorge(bbox, folder="output/"):
    """
    Retrieves trail data within the specified bounding box and saves the trail map as an image and a NumPy array.

    Parameters:
    - bbox (tuple): The bounding box coordinates in the format (minx, miny, maxx, maxy).
    - folder (str): The folder path where the output files will be saved. Default is "output/".

    Returns:
    None
    """

    url = "https://wfs.geonorge.no/skwms1/wfs.turogfriluftsruter"
    feature_types = ["app:Fotrute", "app:AnnenRute", "app:Skiløype", "app:Sykkelrute"]

    bbox_str = create_bbox_string(*bbox)
    gdf_list = []

    print(f"Request to WFS server: https://wfs.geonorge.no/skwms1/wfs.turogfriluftsruter")
    for feature_type in feature_types:
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": feature_type,
            "bbox": bbox_str,
            "outputFormat": "application/gml+xml; version=3.2"
        }
        response = wfs_request(url, params)
        try:
            gdf = gpd.read_file(BytesIO(response.content))
            gdf_list.append(gdf)
        except:
            continue

    if len(gdf_list) == 0:
        print("No trail data found")
        raster = np.zeros((int(bbox[2]-bbox[0]),int(bbox[3]-bbox[1])))

    else:
        combined_df = pd.concat([*gdf_list], ignore_index=True)
        gdf = gpd.GeoDataFrame(combined_df, geometry="geometry")
        #print(gdf.crs)
        gdf = gdf.to_crs(epsg=25833)

        # save gdf plot to file as image
        gdf.plot()
        plt.savefig(f'{folder}trails.png')
        print(f'Trail plot saved to {folder}trails.png')

        minx, miny, maxx, maxy = bbox
        print(bbox)
        width = int(maxx - minx)
        height = int(maxy - miny)
        transform = from_origin(minx, maxy, 1, 1)  # 1x1 meter resolution

        # rasterization
        # 2D NumPy array with 1s for trails and 0s elsewhere
        raster = rasterize(
            [(shape, 1) for shape in gdf.geometry],
            out_shape=(height, width),
            fill=0,
            transform=transform,
            all_touched=True
        )
        

    np.save(f'{folder}array/gn_trail_data.npy', raster)
    print(f'Trail data np array saved to {folder}array/gn_trail_data.npy')



def get_trail_map_osm(utm33_bbox, folder="output/"):
    """
    Retrieves hiking trail data from OpenStreetMap within the specified bounding box
    and saves the trail map as an image and a NumPy array.

    Parameters:
    - bbox (tuple): The bounding box UTM33 format.
    - folder (str): The folder path where the output files will be saved. Default is "output/".

    Returns:
    None
    """
    

    # Convert the bounding box to WGS84 for overpass API
    min_wsg = transform_coords_crs(utm33_bbox[0], utm33_bbox[1], 25833, 4326)
    max_wsg = transform_coords_crs(utm33_bbox[2], utm33_bbox[3], 25833, 4326)
    wsg84_bbox = (min_wsg[0], min_wsg[1], max_wsg[0], max_wsg[1])


    # Overpass API query
    overpass_url = "http://overpass-api.de/api/interpreter"
    bbox_str = f"{wsg84_bbox[1]},{wsg84_bbox[0]},{wsg84_bbox[3]},{wsg84_bbox[2]}"  # min_lat, min_lon, max_lat, max_lon
    print(bbox_str)
    overpass_query = f"""
    [out:json];
    (
    way["highway"="path"]["sac_scale"]({bbox_str});
    relation["route"="hiking"]({bbox_str});
    );
    (._;>;);
    out body;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return

    data = response.json()

    # Process nodes and ways
    nodes = {node['id']: (node['lon'], node['lat']) for node in data['elements'] if node['type'] == 'node'}
    ways = [way for way in data['elements'] if way['type'] == 'way']
    
    # Construct LineStrings from ways
    line_geometries = []
    for way in ways:
        if 'nodes' in way:
            line_coords = [nodes[node_id] for node_id in way['nodes'] if node_id in nodes]
            if line_coords:
                line_geometries.append(LineString(line_coords))
    
    # Check if there are any LineStrings
    if not line_geometries:
        print("No geometries were created. Check Overpass response.")
        raster = np.zeros((int(utm33_bbox[2]-utm33_bbox[0]),int(utm33_bbox[3]-utm33_bbox[1])))
        np.save(f'{folder}array/osm_trail_data.npy', raster)
        return

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=line_geometries)
    gdf.crs = 'EPSG:4326'  # WGS84

    # Convert to UTM (matching projection for rasterization)
    gdf = gdf.to_crs(epsg=25833)

    gdf.plot()
    plt.savefig(f'{folder}/osm_trails.png')
    plt.close()

    # Rasterize
    minx, miny, maxx, maxy = utm33_bbox
    width = int(maxx - minx)
    height = int(maxy - miny)
    transform = from_origin(minx, maxy, 1, 1)  # 1 meter resolution

    raster = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(height, width),
        fill=0,
        transform=transform,
        all_touched=True
    )

    if np.any(raster):
        print("Trails rasterized successfully.")
    else:
        print("Warning: No trails were rasterized. The raster is empty.")

    # Save the raster as a NumPy array
    np.save(f'{folder}array/osm_trail_data.npy', raster)
    print(f'Trail data NumPy array saved to {folder}array/osm_trail_data.npy')


def wfs_request(url, params, retry_count=10):
    """
    Make a request to a Web Feature Service (WFS) and return the response.
    The function automatically retries the request if it fails, up to a specified number of attempts.

    Parameters:
    - params: Dictionary with request parameters.
    - url: String with the WFS service URL.
    - retry_count: Integer with the number of retries for the request.

    Returns:
    - Response object if successful.
    - Raises an Exception if all attempts fail.
    """

    while retry_count > 0:
        try:
            # Make the request
            response = requests.get(url, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                try:
                    # Attempt to parse the XML to ensure it's a valid response
                    ET.fromstring(response.content)
                    return response
                except ET.ParseError:
                    print('Response is not valid XML.')
            else:
                print(f"Request failed with status code: {response.status_code}")

        except requests.RequestException as e:
            print(f"Request failed with exception: {e}")

        retry_count -= 1
        print(f"Retrying request. {retry_count} attempts left.")
    
    raise Exception("Request failed. No attempts left.")




def get_terrain_type_map(center_for_map_squares, start_coords=("00.00","00.00"), rect_radius=1000, folder="output/"):
    """
    Retrieves terrain type map from a WMS server based on the given parameters.

    Args:
        center_for_map_squares (list): List of coordinates representing the center points of map squares.
        start_coords (tuple, optional): Starting coordinates. Defaults to ("00.00","00.00").
        rect_radius (int, optional): Radius of the rectangular area to request from the WMS server. Defaults to 1000.
        folder (str, optional): Output folder path. Defaults to "output/".

    Returns:
        None
    """
    
    url = "https://wms.nibio.no/cgi-bin/ar5?language=nor"

    version = "1.3.0"
    layers = "Arealtype"    
    crs = "EPSG:25833"
    width = rect_radius*2
    height = rect_radius*2
    format = "image/tiff"

    images = []
    tiff_data = []
    futures = []

    with ThreadPoolExecutor() as executor:
        print(f'Request to WMS server: {url}')
        for n, coords in enumerate(center_for_map_squares):
            bbox_coords = calculate_bbox_utm(coords[0], coords[1], rect_radius)
            bbox = f'{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}'

            params = {
            "service": "WMS",
            "request": "GetMap",
            "version": version,
            "layers":  layers,  
            "bbox": bbox,  # (minx, miny, maxx, maxy)
            "crs": crs,  
            "width": width,  
            "height": height,  
            "format": format  
            }
            
            future = executor.submit(wms_request, url, params)
            futures.append(future)
            

        for n, future in enumerate(as_completed(futures)):
            response = future.result()
            if response:
                image = Image.open(BytesIO(response.content))
                images.append(image)
                tiff_data.append(response.content)
                print(f"{n+1}/{len(center_for_map_squares)}")

        # create composite tiff/png and save to file    
        if format == "image/tiff":
            filename = f'{start_coords[0]}_{start_coords[1]}_terrain_composite.tif'
            filepath = f'{folder}{filename}' 
            create_composite_tiff(tiff_data, filepath)
        elif format == "image/png":
            filename = f'{start_coords[0]}_{start_coords[1]}_terrain_composite_image.png'
            filepath = f'{folder}{filename}' 
            create_composite_image(images, filepath)






def wms_request(url, params):
    """
    Make a request to a Web Map Service (WMS) and return the map image.

    Parameters:
    - url: String with the WMS service base URL.
    - params: Dictionary with request parameters such as service, request type, version, layers, bbox, etc.

    Returns:
    - An image object if successful.
    """
    # Make the request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response
    else:
        raise Exception(f"Request failed with status code: {response.status_code}")

