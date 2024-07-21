try:
    from utility import (
        transform_coordinates_to_utm,
        calculate_bbox_utm,
        create_bbox_string,
        extract_tiff_from_multipart_response,
        create_composite_tiff,
        downsample_2d_array,
        matrix_value_padding,
        rasterize_gdf,
        create_composite_image,
        transform_coords_crs,
        write_to_log_file,
        calculate_map_extension
    )
except ImportError:
    from .utility import (
        transform_coordinates_to_utm,
        calculate_bbox_utm,
        create_bbox_string,
        extract_tiff_from_multipart_response,
        create_composite_tiff,
        downsample_2d_array,
        matrix_value_padding,
        rasterize_gdf,
        create_composite_image,
        transform_coords_crs,
        write_to_log_file,
        calculate_map_extension
    )

import requests
from xml.etree import ElementTree as ET
from io import BytesIO
import time
from PIL import Image
from rasterio.transform import from_origin
from rasterio.features import rasterize
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_config import ModelConfig


class ModelGeoDataManager:
    """
    This class contains methods for retrieving and saving various geographical data based on the given parameters.
    
    """


    def __init__(self, config:ModelConfig):
        if not isinstance(config, ModelConfig):
            raise ValueError("The config parameter must be an instance of ModelConfig.")
        self.output_folder = config.OUTPUT_FOLDER
        self.array_folder = config.ARRAY_FOLDER
        self.search_id = config.SEARCH_ID
        self.log_file = config.LOG_FILE
        self.lat = config.LAT
        self.lng = config.LNG
        self.r75 = config.R75
        self.square_radius = config.SQUARE_RADIUS
        self.reduction_factor = config.REDUCTION_FACTOR

        map_size = min(config.R75, config.RADIUS_LIMIT)
        self.map_extention = calculate_map_extension(map_size, config.SQUARE_RADIUS)
        self.full_map_radius = 2*self.map_extention*self.square_radius + self.square_radius
        self.map_squares_center_points = self.calculate_map_square_center_points()
        self.utm33_x, self.utm33_y = transform_coordinates_to_utm(self.lat, self.lng)
        self.utm33_bbox = calculate_bbox_utm(self.utm33_x, self.utm33_y, self.full_map_radius)
        

    def calculate_map_square_center_points(self) -> list:
        """
        Calculate the center points for each map square based on the given latitude and longitude coordinates.
        """
        center_x, center_y = transform_coordinates_to_utm(self.lat, self.lng)
        map_squares_center_point = []
        for y in range(-self.map_extention, self.map_extention+1, 1):
            for x in range(-self.map_extention, self.map_extention+1, 1):
                map_squares_center_point.append((center_x + (2*x*self.square_radius), center_y + (2*y*self.square_radius)))
        return map_squares_center_point

    def get_all_geo_data(self) -> None:
        """
        Retrieves and saves various geo data based on the given latitude and longitude coordinates.
        """

        write_to_log_file(self.log_file, f'Starting data collection for search ID: {self.search_id}\n')
        write_to_log_file(self.log_file, f'Latitude: {self.lat}, Longitude: {self.lng}, r75: {self.r75}m\n')
        start_time = time.perf_counter()


        # get the terrain type map (saves tiff file)
        write_to_log_file(self.log_file, f'Requesting terrain type data...')
        self.get_terrain_type_map()
        write_to_log_file(self.log_file, f' done\n')
    

        # get the height map    (saves tiff file)
        write_to_log_file(self.log_file, f'Requesting terrain height data...')
        self.get_height_map_geonorge()
        write_to_log_file(self.log_file, f' done\n')
    

        # get paths and trails map (saves numpy file)
        write_to_log_file(self.log_file, f'Requesting trail data...')
        self.get_trail_map_geonorge()
        self.get_trail_map_osm()
        write_to_log_file(self.log_file, f' done\n')
    

        # get buildings map (saves numpy file)
        write_to_log_file(self.log_file, f'Requesting building data...')
        self.get_buildings_osm()
        write_to_log_file(self.log_file, f' done\n')
    

        # get railways map (saves numpy file)
        write_to_log_file(self.log_file, f'Requesting railway data...')
        self.get_railways_osm()
        write_to_log_file(self.log_file, f' done\n')

        end_time = time.perf_counter()
        write_to_log_file(self.log_file, f'Data collection done - Time: {end_time-start_time:.2f}\n\n')


    def get_terrain_type_map(self) -> None:
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
        width = self.square_radius*2
        height = self.square_radius*2
        format = "image/tiff"

        images = []
        tiff_data = []
        futures = []

        with ThreadPoolExecutor() as executor:
            print(f'Request to WMS server: {url}')
            for n, coords in enumerate(self.map_squares_center_points):
                bbox_coords = calculate_bbox_utm(coords[0], coords[1], self.square_radius)
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
                
                future = executor.submit(self._wms_request, url, params)
                futures.append(future)
                

            for n, future in enumerate(as_completed(futures)):
                response = future.result()
                if response:
                    image = Image.open(BytesIO(response.content))
                    images.append(image)
                    tiff_data.append(response.content)
                    print(f"{n+1}/{len(self.map_squares_center_points)}")

                    if n == 0:
                        with open(self.log_file, 'a') as f:
                            f.write(f'{n+1}/{len(self.map_squares_center_points)}')
                    else:
                        if n == 9 or n == 99 or n == 999:
                            with open(self.log_file, 'a') as f:
                                f.write(f' ')
                        with open(self.log_file, 'rb+') as f:
                            f.seek(0, 2)  # Move to the end of the file
                            file_size = f.tell()
                            text = f'{n+1}/{len(self.map_squares_center_points)}'
                            f.seek(max(0, file_size - len(text)), 0)  # Move pointer back
                            f.write(text.encode())  # Write the updated percentage

            # create composite tiff/png and save to file    
            if format == "image/tiff":
                filename = f'id{self.search_id}_{self.lat}_{self.lng}_terrain_composite.tif'
                filepath = f'{self.output_folder}{filename}' 
                create_composite_tiff(tiff_data, filepath)
            elif format == "image/png":
                filename = f'id{self.search_id}_{self.lat}_{self.lng}_terrain_composite_image.png'
                filepath = f'{self.output_folder}{filename}' 
                create_composite_image(images, filepath)


    def get_height_map_geonorge(self) -> None:
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
        min_size_expected = 4*(2*self.square_radius**2)
        print(f'Minimum size expected: {min_size_expected} bytes')

        with open(self.log_file, 'a') as f:
            f.write(f'{0}/{len(self.map_squares_center_points)}') 

        with ThreadPoolExecutor() as executor:
            print(f'Request to WCS: {url}')
            #for n, coords in reversed(list(enumerate(center_of_map_squares))):
            for n, coords in enumerate(self.map_squares_center_points):  
                bbox_coords = calculate_bbox_utm(coords[0], coords[1], self.square_radius)
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
                future = executor.submit(self._wcs_request, url, params, min_size_limit=min_size_expected, n=n)
                futures.append(future)
                print(f'Request {n+1} submitted. Waiting for response...')
                time.sleep(0.5)

            # Wait for all futures to complete
            for n, future in enumerate(as_completed(futures)):
                response = future.result()
                if response:
                    tiff_data.append(extract_tiff_from_multipart_response(response))

                    
                    if n == 0:
                        pass
                        # with open(log_file, 'a') as f:
                        #     f.write(f'{n+1}/{len(center_of_map_squares)}') 
                    else:
                        if n == 9 or n == 99 or n == 999:
                            with open(self.log_file, 'a') as f:
                                f.write(f' ')
                        with open(self.log_file, 'rb+') as f:
                            f.seek(0, 2)  # Move to the end of the file
                            file_size = f.tell()
                            text = f'{n+1}/{len(self.map_squares_center_points)}'
                            f.seek(max(0, file_size - len(text)), 0)  # Move pointer back
                            f.write(text.encode())  # Write the updated percentage

        # combine a composite tiff from all tiff data and save to file
        filename = f'id{self.search_id}_{self.lat}_{self.lng}_height_composite.tif' 
        create_composite_tiff(tiff_data, f'{self.output_folder}{filename}')


    def get_trail_map_geonorge(self) -> None:
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

        bbox_str = create_bbox_string(*self.utm33_bbox)
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
            response = self._wfs_request(url, params)
            try:
                gdf = gpd.read_file(BytesIO(response.content))
                gdf_list.append(gdf)
            except:
                continue

        if len(gdf_list) == 0:
            print("No trail data found")
            raster = np.zeros((int(self.utm33_bbox[2]-self.utm33_bbox[0]),int(self.utm33_bbox[3]-self.utm33_bbox[1])))

        else:
            combined_df = pd.concat([*gdf_list], ignore_index=True)
            gdf = gpd.GeoDataFrame(combined_df, geometry="geometry")
            gdf = gdf.to_crs(epsg=25833)

            minx, miny, maxx, maxy = self.utm33_bbox
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

        # Padding trails    
        raster = matrix_value_padding(raster, 1, 20)
        # Downsample the raster 
        raster = downsample_2d_array(raster, self.reduction_factor)

        np.save(f'{self.array_folder}id{self.search_id}_gn_trail_data.npy', raster)
        print(f'Trail data np array saved to {self.array_folder}{self.search_id}_gn_trail_data.npy')


    def get_trail_map_osm(self):
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
        min_wsg = transform_coords_crs(self.utm33_bbox[0], self.utm33_bbox[1], 25833, 4326)
        max_wsg = transform_coords_crs(self.utm33_bbox[2], self.utm33_bbox[3], 25833, 4326)
        wsg84_bbox = (min_wsg[0], min_wsg[1], max_wsg[0], max_wsg[1])

        # Overpass API query
        overpass_url = "http://overpass-api.de/api/interpreter"
        bbox_str = f"{wsg84_bbox[1]},{wsg84_bbox[0]},{wsg84_bbox[3]},{wsg84_bbox[2]}"  # min_lat, min_lon, max_lat, max_lon
        print(f'Request to Overpass API for trails: {overpass_url}')
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
            raster = np.zeros((int(self.utm33_bbox[2]-self.utm33_bbox[0]),int(self.utm33_bbox[3]-self.utm33_bbox[1])))
            np.save(f'{self.output_folder}array/id{self.search_id}_osm_trail_data.npy', raster)
            return

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=line_geometries)
        gdf.crs = 'EPSG:4326'  # WGS84

        # Convert to UTM (matching projection for rasterization)
        gdf = gdf.to_crs(epsg=25833)

        # Rasterize
        minx, miny, maxx, maxy = self.utm33_bbox
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

        # Padding trails
        raster = matrix_value_padding(raster, 1, 20)
        # Downsample the raster
        raster = downsample_2d_array(raster, self.reduction_factor)

        # Save the raster as a NumPy array
        np.save(f'{self.array_folder}id{self.search_id}_osm_trail_data.npy', raster)
        print(f'Trail data NumPy array saved to {self.array_folder}osm_trail_data.npy')


    def get_buildings_osm(self) -> None:
        """
        Retrieves building data from the OpenStreetMap Overpass API within a given bounding box,
        rasterizes the buildings, and saves the resulting raster as a NumPy array.

        Args:
            utm33_bbox (tuple): The bounding box coordinates in UTM Zone 33N projection.
            folder (str, optional): The folder path to save the output files. Defaults to "output/".
            reduction_factor (int, optional): The factor by which to downsample the raster. Defaults to 5.
            search_id (int, optional): The ID of the search. Defaults to 0.
        """
        
        # Convert the bounding box to WGS84 for overpass API
        min_wsg = transform_coords_crs(self.utm33_bbox[0], self.utm33_bbox[1], 25833, 4326)
        max_wsg = transform_coords_crs(self.utm33_bbox[2], self.utm33_bbox[3], 25833, 4326)
        wsg84_bbox = (min_wsg[0], min_wsg[1], max_wsg[0], max_wsg[1])

        # Overpass API query
        overpass_url = "http://overpass-api.de/api/interpreter"
        bbox_str = f"{wsg84_bbox[1]},{wsg84_bbox[0]},{wsg84_bbox[3]},{wsg84_bbox[2]}"  # min_lat, min_lon, max_lat, max_lon
        print(f'Request to Overpass API for buildings: {overpass_url}')
        overpass_query = f"""
        [out:json];
        (
        way["building"]({bbox_str});
        relation["building"]({bbox_str});
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
            raster = np.zeros((int(self.utm33_bbox[2]-self.utm33_bbox[0]),int(self.utm33_bbox[3]-self.utm33_bbox[1])))
            np.save(f'{self.output_folder}array/id{self.fsearch_id}_osm_building_data.npy', raster)
            return

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=line_geometries)
        gdf.crs = 'EPSG:4326'  # WGS84

        # Convert to UTM (matching projection for rasterization)
        gdf = gdf.to_crs(epsg=25833)

        # Rasterize
        minx, miny, maxx, maxy = self.utm33_bbox
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
            print("Buildings rasterized successfully.")
        else:
            print("Warning: No buildings were rasterized. The raster is empty.")


        # Padding buildings
        raster = matrix_value_padding(raster, 1, 2)
        # Downsample the raster
        raster = downsample_2d_array(raster, self.reduction_factor)

        # Save the raster as a NumPy array
        np.save(f'{self.array_folder}id{self.search_id}_osm_building_data.npy', raster)
        print(f'Building data NumPy array saved to {self.array_folder}osm_buildings_data.npy')


    def get_railways_osm(self) -> None:
        """
        Retrieves railway data from the OpenStreetMap (OSM) API within the specified bounding box,
        rasterizes the railways, and saves the raster as a NumPy array.

        Args:
            utm33_bbox (tuple): The bounding box coordinates in UTM33 projection (minx, miny, maxx, maxy).
            folder (str, optional): The folder path to save the NumPy array. Defaults to "output/".
            reduction_factor (int, optional): The reduction factor for downsampling the raster. Defaults to 5.
            search_id (int, optional): The search ID for identifying the railway data. Defaults to 0.

        Returns:
            None
        """
        
        # Convert the bounding box to WGS84 for overpass API
        min_wsg = transform_coords_crs(self.utm33_bbox[0], self.utm33_bbox[1], 25833, 4326)
        max_wsg = transform_coords_crs(self.utm33_bbox[2], self.utm33_bbox[3], 25833, 4326)
        wsg84_bbox = (min_wsg[0], min_wsg[1], max_wsg[0], max_wsg[1])

        print(f'Request to Overpass API for railways: http://overpass-api.de/api/interpreter')

        # Overpass API query
        railway_query = f"""
            [out:json];
            (
            // Collect railway crossings
            node["railway"="level_crossing"]({wsg84_bbox[1]},{wsg84_bbox[0]},{wsg84_bbox[3]},{wsg84_bbox[2]});
            //node["railway"="crossing"]({wsg84_bbox[1]},{wsg84_bbox[0]},{wsg84_bbox[3]},{wsg84_bbox[2]});
            )->.crossings;

            // Collect railways that are not in tunnels or on bridges
            way["railway"="rail"]["tunnel"!="yes"]["bridge"!="yes"]({wsg84_bbox[1]},{wsg84_bbox[0]},{wsg84_bbox[3]},{wsg84_bbox[2]});
        
            // Remove ways that intersect with any of the crossing nodes
            (._; - way(around.crossings:0););

            (._;>;);
            out body;

        """

        road_query = f"""
            [out:json];
            (
            way["highway"]["bridge"="yes"]({wsg84_bbox[1]},{wsg84_bbox[0]},{wsg84_bbox[3]},{wsg84_bbox[2]});
            // Fetch ways that are tagged as tunnels
            way["highway"]["tunnel"="yes"]({wsg84_bbox[1]},{wsg84_bbox[0]},{wsg84_bbox[3]},{wsg84_bbox[2]});
            );
            (._;>;);
            out body;
        """

        railway_response = self._overpass_request(railway_query)
        road_response = self._overpass_request(road_query)

        # Construct LineStrings from ways
        road_line_geometries = self._get_line_geometries_from_overpass_response(road_response)
        railway_line_geometries = self._get_line_geometries_from_overpass_response(railway_response)
    
        # Rasterize boundaries
        minx, miny, maxx, maxy = self.utm33_bbox
        width = int(maxx - minx)
        height = int(maxy - miny)
        transform = from_origin(minx, maxy, 1, 1)  # 1 meter resolution

        if not railway_line_geometries:
            print("No railway raster in railways.")
            railway_raster = np.zeros((int(self.utm33_bbox[2]-self.utm33_bbox[0]),int(self.utm33_bbox[3]-self.utm33_bbox[1])))
        else:
            # Create GeoDataFrame
            railway_gdf = gpd.GeoDataFrame(geometry=railway_line_geometries)
            railway_gdf.crs = 'EPSG:4326'  # WGS84
            railway_gdf = railway_gdf.to_crs(epsg=25833)

            railway_raster = rasterize_gdf(railway_gdf, height, width, transform)
        
        if not road_line_geometries:
            print("No road raster in railways.")
            road_raster = np.zeros((int(self.utm33_bbox[2]-self.utm33_bbox[0]),int(self.utm33_bbox[3]-self.utm33_bbox[1])))
        else:
            # Create GeoDataFrame
            road_gdf = gpd.GeoDataFrame(geometry=road_line_geometries)
            road_gdf.crs = 'EPSG:4326'  # WGS84
            road_gdf = road_gdf.to_crs(epsg=25833)

            road_raster = rasterize_gdf(road_gdf, height, width, transform)
            matrix_value_padding(road_raster, 1, 10)    # padding roads

        railway_raster[road_raster == 1] = 0    # remove crossing roads on bridge/tunnels from railways

        if np.any(railway_raster):
            print("Railways rasterized successfully.")
        else:
            print("Warning: No railways were rasterized. The raster is empty.")

        # Padding railways
        railway_raster = matrix_value_padding(railway_raster, 1, 8)
        # Downsample the raster
        railway_raster = downsample_2d_array(railway_raster, self.reduction_factor)

        # Save the raster as a NumPy array
        np.save(f'{self.array_folder}id{self.search_id}_osm_railway_data.npy', railway_raster)
        print(f'Railway data NumPy array saved to {self.array_folder}osm_railway_data.npy')


    def _get_line_geometries_from_overpass_response(self, data) -> list:
        nodes = {node['id']: (node['lon'], node['lat']) for node in data['elements'] if node['type'] == 'node'}
        ways = [way for way in data['elements'] if way['type'] == 'way']

        # Construct LineStrings from ways
        line_geometries = []
        for way in ways:
            if 'nodes' in way:
                line_coords = [nodes[node_id] for node_id in way['nodes'] if node_id in nodes]
                if line_coords:
                    line_geometries.append(LineString(line_coords))
        return line_geometries


    def _wcs_request(self, url, params, retry_limit=15, min_size_limit=100000, n=0) -> requests.Response:
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
        retry_count = 0
        while retry_limit > retry_count:
            # Make the request
            response = requests.get(url, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type')
                if 'multipart' in content_type: # Check if the response is multipart xml
                    
                    
                    if len(response.content) < min_size_limit and retry_count <= retry_limit-5:
                        print(f'Error Request {n+1}: Insufficient data. {len(response.content)}/{min_size_limit} bytes')
                    elif len(response.content) < min_size_limit/4 and retry_count <= retry_limit-2:
                        print(f'Error Request {n+1}: Insufficient data. {len(response.content)}/{min_size_limit/4} bytes')

                    else:
                        print(f'Request {n+1} successful. Count: {len(response.content)} bytes')
                        return response
                    
                else:
                    print('Response unsuccessful. Content-Type is not multipart.')  # Unexpected response
                    time.sleep(2) 
                    continue
            else:
                print(f"Request failed with status code: {response.status_code}")

            retry_count += 1
            print(f"Retrying request {n+1}. Attempts left: {retry_limit-retry_count}")
            if retry_count < retry_limit - 5:
                time.sleep(1)
            else:
                time.sleep(5)

        raise Exception("Request failed. No attempts left.")



    def _wms_request(self, url, params) -> requests.Response:
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
        


    def _wfs_request(self, url, params, retry_count=10) -> requests.Response:
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

    
    def _overpass_request(self, overpass_query) -> dict:
        # Overpass API query
        overpass_url = "http://overpass-api.de/api/interpreter"

        response = requests.get(overpass_url, params={'data': overpass_query})
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            return

        data = response.json()
        return data