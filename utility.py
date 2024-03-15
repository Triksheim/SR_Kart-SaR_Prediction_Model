import pyproj
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



def transform_coordinates_to_utm(lat, lng):
    """
    Convert a lat-lng coordinate to UTM33N.
    
    Parameters:
    - lat (float): Latitude of the point.
    - lng (float): Longitude of the point.
    
    Returns:
    - utm_x (float): UTM X-coordinate of the point.
    - utm_y (float): UTM Y-coordinate of the point.
    """
    
    # Define the source CRS (WGS84) and target CRS (EPSG:25833)
    wgs84 = pyproj.CRS('EPSG:4326')
    utm33n = pyproj.CRS('EPSG:25833')

    # Create a transformer to convert between CRS
    transformer = pyproj.Transformer.from_crs(wgs84, utm33n, always_xy=True)
    
    # Convert the point to the target CRS
    utm_x, utm_y = transformer.transform(lng, lat)
    
    return utm_x, utm_y


def calculate_bbox(utm_x, utm_y, radius=1000):
    """
    Calculate a bounding box from a UTM coordinate with a specified radius.
    
    Parameters:
    - utm_x (float): UTM X-coordinate of the center point.
    - utm_y (float): UTM Y-coordinate of the center point.
    - radius (float): Radius in meters for the bbox.
    
    Returns:
    - bbox (tuple): A tuple representing the bbox (min_x, min_y, max_x, max_y) in EPSG:25833.
    """
    
    # Calculate bbox coordinates based on the radius
    min_x = utm_x - radius
    min_y = utm_y - radius
    max_x = utm_x + radius
    max_y = utm_y + radius
    
    return (min_x, min_y, max_x, max_y)

def create_bbox_string(min_x, min_y, max_x, max_y, crs='EPSG::25833'):
    """
    Create a bounding box string for a WCS GetCoverage request.
    
    Parameters:
    - min_x (float): Minimum x-coordinate of the bbox.
    - min_y (float): Minimum y-coordinate of the bbox.
    - max_x (float): Maximum x-coordinate of the bbox.
    - max_y (float): Maximum y-coordinate of the bbox.
    - crs (str): The coordinate reference system (CRS) of the bbox.
    
    Returns:
    - bbox_string (str): A string representing the bbox in the format "min_x,min_y,max_x,max_y,crs".
    """
    return f"{min_x},{min_y},{max_x},{max_y},urn:ogc:def:crs:{crs}"


def latlng_to_utm_bbox(lat, lng, radius):
    """
    Convert a lat-lng coordinate to a bbox in EPSG:25833 with a specified radius.
    
    Parameters:
    - lat (float): Latitude of the center point.
    - lng (float): Longitude of the center point.
    - radius (float): Radius in meters for the bbox.
    
    Returns:
    - bbox (tuple): A tuple representing the bbox (min_x, min_y, max_x, max_y) in EPSG:25833.
    """
    
    # Define the source CRS (WGS84) and target CRS (EPSG:25833)
    wgs84 = pyproj.CRS('EPSG:4326')
    utm33n = pyproj.CRS('EPSG:25833')

    # Create a transformer to convert between CRS
    transformer = pyproj.Transformer.from_crs(wgs84, utm33n, always_xy=True)
    
    # Convert the center point to the target CRS
    center_x, center_y = transformer.transform(lng, lat)
    print(center_x, center_y, "2")
    
    # Calculate bbox coordinates based on the radius
    min_x = center_x - radius
    min_y = center_y - radius
    max_x = center_x + radius
    max_y = center_y + radius
    
    return (min_x, min_y, max_x, max_y)


def extract_tiff_from_multipart_response(response, output_path="", save_to_file=False):
    """
    Extract a GeoTIFF from a multipart response and save it to a file.
    
    Parameters:
    - response (requests.Response): The response object from the GetCoverage request.
    - output_path (str): The path to save the GeoTIFF file.
    - save_to_file (bool): Whether to save the GeoTIFF to a file or not. Returns the tiff_data when False.
    """
    
    boundary = b'--wcs'
    start_of_tiff_part = response.content.find(boundary + b'\nContent-Type: image/tiff')

    if start_of_tiff_part != -1:
        start_of_data = response.content.find(b'\n\n', start_of_tiff_part) + 2  # Assuming double newline marks end of headers
        end_of_tiff_part = response.content.find(boundary, start_of_data)
        tiff_data = response.content[start_of_data:end_of_tiff_part].strip()
        
        if save_to_file:
            # Save the TIFF data to a file
            with open(output_path, 'wb') as tiff_file:
                tiff_file.write(tiff_data)
            print("TIFF file extracted and saved.")
        else:
            return tiff_data

    else:
        print("TIFF part not found in response.")


def exctract_data_from_tiff(tiff_path, band_n=1, tiff_data=None):
    """
    Read a GeoTIFF file single band and return the data as a numpy array.
    
    Parameters:
    - tiff_path (str): The path to the GeoTIFF file.
    - tiff_data (bytes): The TIFF data as bytes. If data is provied, tiff_path is ignored.
    
    Returns:
    - data (numpy.ndarray): The data as a 2D numpy array.
    """
    if tiff_path is None and tiff_data is None:
        raise ValueError("Either tiff_path or tiff_data must be provided.")
    
    if tiff_data is None:
        with rasterio.open(tiff_path) as src:
            data = src.read(band_n)   # band mumber
    else:
        with rasterio.io.MemoryFile(tiff_data) as memfile:
            with memfile.open() as src:
                data = src.read(band_n)  # band number
    return data



def create_composite_tiff(tiff_data, output_path):
    """
    Create a composite TIFF from in-memory TIFF data.

    Parameters:
    - tiff_data (list): A list of in-memory TIFF data (bytes).
    - output_path (str): Path to save the composite TIFF file.
    """
    src_files_to_mosaic = []
    memory_files = []

    # Open each TIFF data in a MemoryFile and keep them open
    for data in tiff_data:
        memfile = MemoryFile(data)
        dataset = memfile.open()
        src_files_to_mosaic.append(dataset)
        memory_files.append(memfile)  # Keep a reference to prevent closing

    # Merge datasets into a single composite array
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Create metadata for the composite image
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        #"compress": "lzw"  # Apply LZW compression
    })

    # Write the composite array to a new TIFF file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        print(f"Composite TIFF saved to: {output_path}")

    # Clean up: Close all MemoryFiles and datasets
    for dataset in src_files_to_mosaic:
        dataset.close()
    for memfile in memory_files:
        memfile.close()


def calculate_image_size(images):
    """Calculate the total size of the composite image based on individual images."""
    # Assuming all images are the same size and the grid is square
    if images:
        image_count = len(images)
        grid_size = int(image_count ** 0.5)  # Square root to find grid dimensions
        single_width, single_height = images[0].size
        return single_width * grid_size, single_height * grid_size
    else:
        return 0, 0


def create_composite_image(images, output_path):
    """Create a composite image from a list of images."""
    # Calculate the size of the composite image
    composite_width, composite_height = calculate_image_size(images)
    composite_image = Image.new('RGB', (composite_width, composite_height))

    # Assuming all images are the same size
    single_width, single_height = images[0].size
    grid_size = int(len(images) ** 0.5)  # Assuming a square grid

    for index, image in enumerate(images):
        # Calculate the position of the current image in the grid
        grid_x = index % grid_size
        grid_y = index // grid_size

        # Reverse the y-coordinate to start from the top
        grid_y = (len(images) // grid_size - 1) - grid_y

        # Calculate the position where the current image should be pasted
        paste_x = grid_x * single_width
        paste_y = grid_y * single_height

        # Paste the current image into the composite image
        composite_image.paste(image, (paste_x, paste_y))

    composite_image.save(output_path)
    print(f"Composite image saved to: {output_path}")



def reduce_resolution(data, factor=10, method="mean"):
    if method == "mean":
        return data.reshape(data.shape[0]//factor, factor, data.shape[1]//factor, factor).mean(axis=1).mean(axis=2)
    elif method == "max":
        return data.reshape(data.shape[0]//factor, factor, data.shape[1]//factor, factor).max(axis=1).max(axis=2)
    elif method == "min":
        return data.reshape(data.shape[0]//factor, factor, data.shape[1]//factor, factor).mean(axis=1).min(axis=2)

def calc_steepness(height_data):
    # Calculate gradients along both axes
    grad_x, grad_y = np.gradient(height_data)
    # Calculate the steepness/elevation change
    steepness = np.sqrt(grad_x**2 + grad_y**2)
    # steepness now represents the elevation change or steepness for each square
    return steepness

def combine_matrixes(terrain, steepness, method="mean"):
    if terrain.shape != steepness.shape:
        print("Matrixes are not the same size")
        return
    
    if method == "mean":
        return (terrain + steepness) / 2
    
    elif method == "multiply":
        return terrain * steepness
    
    elif method == "square":
        return terrain * steepness**2
    

def plot_array(array, cmap="terrain", label=""):
    plt.imshow(array, cmap=cmap)
    plt.colorbar(label=label)
    plt.axis('off')
    plt.show()