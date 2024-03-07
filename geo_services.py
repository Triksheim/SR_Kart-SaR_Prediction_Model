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
            "version": "1.1.1",
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
            if 'multipart' in content_type: # contains tiff image
                if len(response.content) > 10000:
                    return response
                print(f'Error:Insufficient data. Count: {len(response.content)} bytes')
                
            else:
                print('Response unsuccessful. Content-Type is not multipart.')   
        else:
            print(f"Request failed with status code: {response.status_code}")

        retry_count -= 1
        print(f"Retrying request. Attempts left: {retry_count}")
        time.sleep(2)

    raise Exception("Request failed. No attempts left.")


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


def wms_example():
    # Example use
    params = {
    "service": "WMS",
    "request": "GetMap",
    "version": "1.3.0",
    "layers": "Arealtype",  # Specify the layer(s) you want to retrieve
    "bbox": "601128.9434,7593652.4896,601648.8557,7594129.2192",  # Define the bounding box (minx, miny, maxx, maxy)
    "crs": "EPSG:25833",  # Specify the coordinate reference system
    "width": "2000",  # Image width
    "height": "2000",  # Image height
    "format": "image/png"  # Specify the output format
    }
    url = "https://wms.nibio.no/cgi-bin/ar5?language=nor"

    # Make the request
    map_image = wms_request(url, params)

    # You can now display, save or further process the map image
    map_image.show()



def wfs_example():
    # Example use
    url = "https://wfs.geonorge.no/skwms1/wfs.turogfriluftsruter"  # Adjust the URL based on the actual endpoint
    # params = {
    #     "service": "WFS",
    #     "request": "GetFeature",
    #     "version": "2.0.0",
    #     "typeNames": "app:Fotrute",  # Adjust based on the specific feature type you need
    #     "bbox": "586855.4027,7585022.1894,617296.7404,7603233.4645,urn:ogc:def:crs:EPSG::25833",
    #     "outputFormat": "application/gml+xml; version=3.2"
    # }
    
    # # Make the request
    # response = get_features(params, url)
    # data = response.content
    # #print(response.content[:10000])


    feature_types = ["app:Fotrute", "app:AnnenRute", "app:Skiløype", "app:Sykkelrute"]
    gdf_list = []

    for feature_type in feature_types:
        params = {
            "service": "WFS",
            "request": "GetFeature",
            "version": "2.0.0",
            "typeNames": feature_type,
            "bbox": "583471.5080,7582958.8789,616620.8362,7602612.4136,urn:ogc:def:crs:EPSG::25833",
            "outputFormat": "application/gml+xml; version=3.2"
        }
        response = wfs_request(params, url)
        try:
            gdf = gpd.read_file(BytesIO(response.content))
            gdf_list.append(gdf)
        except:
            print(f"Failed to read feature type: {feature_type}")

    combined_df = pd.concat([*gdf_list], ignore_index=True)
    gdf = gpd.GeoDataFrame(combined_df, geometry="geometry")



    # gml_data = BytesIO(data)
    # gdf = read_file(gml_data)

    print(gdf.crs)
    gdf = gdf.to_crs(epsg=25833)
    print(gdf.crs)
    
    # Plot the footpaths
    gdf.plot()
    # Customize the plot
    plt.title('Trails')
    # Show the plot
    plt.show()

    # Bounds in EPSG:25833 - directly from your WFS request bbox
    minx, miny, maxx, maxy = 583471.5080,7582958.8789,616620.8362,7602612.4136
    width = int(maxx - minx)
    height = int(maxy - miny)

    # Define the transform
    transform = from_origin(minx, maxy, 1, 1)  # 1x1 meter resolution

    # Proceed with rasterization as before
    raster = rasterize(
        [(shape, 1) for shape in gdf.geometry],
        out_shape=(height, width),
        fill=0,
        transform=transform,
        all_touched=True
    )

    # raster is now a 2D NumPy array with 1s for footpaths and 0s elsewhere
    print(raster.shape)
    print(np.unique(raster, return_counts=True))

        # Here you can process the response, e.g., parse the XML, extract features, etc.

if __name__ == "__main__":
    wms_example()
    wfs_example()
 