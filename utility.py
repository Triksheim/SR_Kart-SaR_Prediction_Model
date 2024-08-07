"""
This Python module provides a suite of tools for geographical data manipulation and analysis, particularly focusing on coordinate transformations, geospatial raster and vector data processing, and visualization. Key functionalities include:

- Transforming latitude and longitude to UTM coordinates and vice versa.
- Calculating bounding boxes based on UTM coordinates.
- Creating bounding box strings for web coverage service (WCS) requests.
- Converting geographic coordinates to bounding boxes with specified radii.
- Extracting GeoTIFF data from multipart responses and performing operations on them such as reading, merging, and creating composite TIFF images.
- Manipulating and analyzing raster data using Rasterio and creating visualizations with Matplotlib and Pillow.
- Performing complex geospatial operations with Shapely and GeoPandas, such as calculating alpha shapes, creating convex hulls, and generating map overlays.

Note:
The module is designed to be used in the SAR (Search and Rescue) application SR Kart created during bachelor's thesis at UiT The Arctic University of Norway.
The tools provided here are intended to support the development of a web-based application for search and rescue operations in Norway,
focusing on the use of geospatial data to optimize search strategies and improve the efficiency of rescue missions.

Dependencies:
- pyproj
- rasterio
- PIL
- matplotlib
- numpy
- shapely
- scipy
- geopandas

Author: Martin Riksheim
Date: 2024
"""




import pyproj
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from shapely.geometry import Polygon, MultiLineString, MultiPolygon,  GeometryCollection
from shapely.ops import polygonize, unary_union, nearest_points
from scipy.spatial import Delaunay
import geopandas as gpd
from rasterio.features import rasterize





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


def transform_coords_crs(x, y, source_crs, target_crs):
    new_crs = pyproj.CRS(target_crs)
    old_crs = pyproj.CRS(source_crs)

    # Create a transformer to convert between CRS
    transformer = pyproj.Transformer.from_crs(old_crs, new_crs, always_xy=True)
    
    # Convert the point to the target CRS
    new_coords = transformer.transform(x, y)
    
    return new_coords


def calculate_bbox_utm(utm_x, utm_y, radius=1000):
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
    Extract a GeoTIFF from a multipart response (GeoNorge spesific) and save it to a file.
    
    Parameters:
    - response (requests.Response): The response object from the GetCoverage request.
    - output_path (str): The path to save the GeoTIFF file.
    - save_to_file (bool): Whether to save the GeoTIFF to a file or not. Returns the tiff_data when False.
    """
    #print(f'{len(response.content)} bytes received.')
    boundary = b'--wcs'
    start_of_tiff_part = response.content.find(boundary + b'\nContent-Type: image/tiff')

    if start_of_tiff_part != -1:
        start_of_data = response.content.find(b'\n\n', start_of_tiff_part) + 2  # Assuming double newline marks end of headers
        end_of_tiff_part = response.content.find(boundary, start_of_data)
        tiff_data = response.content[start_of_data:end_of_tiff_part].strip()
        #print("TIFF part found:", len(tiff_data))
        
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
    """
    Create a composite image from a list of images.
    """
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


def create_numpy_arrays_from_tiff(search_id, start_lat, start_lng, tiff_folder, output_folder):
    # convert height tiff to numpy array and save to file
    create_height_array(f'{tiff_folder}id{search_id}_{start_lat}_{start_lng}_height_composite.tif', output_folder)

    # convert terrain tiff to 3d numpy array with RGB values and save to file
    create_terrain_RGB_array(f'{tiff_folder}id{search_id}_{start_lat}_{start_lng}_terrain_composite.tif', output_folder)


def create_height_array(tiff_path, output_folder="output/array", reduction_factor=5, search_id=0):
    """
    Create a NumPy array from a TIFF file containing height data.

    Parameters:
    filepath (str): The path to the TIFF file.
    folder (str, optional): The folder to save the NumPy array. Default is "output/".

    Returns:
    None
    """

    height_dataset = exctract_data_from_tiff(tiff_path)
    height_dataset = height_dataset[:-1,:-1]  # Remove last row and column to make it square  
    height_dataset = reduce_resolution(height_dataset, factor=reduction_factor, method='mean')

    np.save(f'{output_folder}id{search_id}_height_matrix.npy', height_dataset)
    print(f'Height data np array saved to {output_folder}height_matrix.npy')



def create_terrain_RGB_array(filepath, output_folder="output/array/", reduction_factor=5, search_id=0):
    """
    Create a RGB array from a terrain dataset stored in a TIFF file.

    Args:
        filepath (str): The path to the TIFF file.
        folder (str, optional): The folder to save the RGB array. Defaults to "output/array/".

    Returns:
        None
    """
    terrain_dataset_R = exctract_data_from_tiff(tiff_path=filepath, band_n=1)
    terrain_dataset_G = exctract_data_from_tiff(tiff_path=filepath, band_n=2)
    terrain_dataset_B = exctract_data_from_tiff(tiff_path=filepath, band_n=3)
    terrain_dataset = np.array([terrain_dataset_R, terrain_dataset_G, terrain_dataset_B])

    # Downsample the terrain dataset
    terrain_dataset = downsample_rgb_image(terrain_dataset, factor=reduction_factor)

    np.save(f'{output_folder}id{search_id}_terrain_RGB_matrix.npy', terrain_dataset)
    print(f'Terrain RGB data np array saved to {output_folder}terrain_RGB_matrix.npy')



def plot_array(array, cmap="terrain", label="", title="Array Plot",colorbar=True, save=False, folder="output/"):
    plt.imshow(array, cmap=cmap)
    if colorbar:
        plt.colorbar(label=label)
    plt.title(title)
    plt.axis('off')
    if save:
        plt.savefig(f'{folder}{title}.png')
        plt.close()
    else:
        plt.show()


def compute_concave_hull_from_points(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.
    @param points: Iterable container of points.
    @param alpha: Alpha value to influence the gooeyness of the border. Smaller numbers
                  don't fall inward as much as larger numbers. Too large, and you lose everything!
    """
    # Function based on: https://gist.github.com/jclosure/d93f39a6c7b1f24f8b92252800182889 

    if len(points) < 4:
        # A polygon cannot be made with fewer than 3 points
        return None

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    tri = Delaunay(points)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Lengths of sides of triangle
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # radius filter
        if circum_r < alpha:
            add_edge(edges, edge_points, points, ia, ib)
            add_edge(edges, edge_points, points, ib, ic)
            add_edge(edges, edge_points, points, ic, ia)
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    concave_hull = unary_union(triangles)
    
    #print(f'First:{concave_hull.geom_type=}')

    # Ensure the result is a single polygon
    if concave_hull.geom_type == 'GeometryCollection' or isinstance(concave_hull, MultiPolygon):
        concave_hull = connect_polygons_with_thin_polygons(concave_hull)
        #concave_hull = iterative_merge_close_polygons(concave_hull)

    #print(f'End: {concave_hull.geom_type=}')

    return concave_hull


def connect_polygons_with_thin_polygons(multipolygon, width=1, extension=1):
    """
    Connects the polygons in a MultiPolygon with thin connecting polygons.

    Args:
        multipolygon (MultiPolygon): The MultiPolygon containing the polygons to connect.
        width (float, optional): The width of the connecting polygons. Defaults to 1.
        extension (float, optional): The extension of the connecting polygons. Defaults to 1.

    Returns:
        Polygon or MultiPolygon: The combined polygon or multipolygon after connecting the polygons.
    """

    if not isinstance(multipolygon, MultiPolygon):
        return multipolygon
    
    polygons = list(multipolygon.geoms)
    combined = polygons[0]
    for poly in polygons[1:]:
        # Find the nearest points between the two polygons
        p1, p2 = nearest_points(combined, poly)
        
        if p1.equals(p2):
            # Skip if the nearest points are the same
            continue
        
        # Create a thin connecting polygon (rectangular strip)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            # If length is zero, skip to avoid division by zero
            continue
        
        offset_x = width * dy / length / 2
        offset_y = width * dx / length / 2
        ex_x = extension * dx / length
        ex_y = extension * dy / length
        thin_polygon = Polygon([
            (p1.x - offset_x - ex_x, p1.y + offset_y - ex_y),
            (p1.x + offset_x - ex_x, p1.y - offset_y - ex_y),
            (p2.x + offset_x + ex_x, p2.y - offset_y + ex_y),
            (p2.x - offset_x + ex_x, p2.y + offset_y + ex_y),
        ])
        
        # Combine the current combined polygon with the new polygon and the connecting thin polygon
        combined = unary_union([combined, poly, thin_polygon])
    
    # Handle the case where the result is a GeometryCollection
    if isinstance(combined, GeometryCollection):
        polygons = [geom for geom in combined.geoms if isinstance(geom, Polygon)]
        if polygons:
            combined = unary_union(polygons)
    
    return combined



def iterative_merge_close_polygons(multipolygon, max_buffer_distance=1.0, step=0.1):
    # Testing. Not used.
    if isinstance(multipolygon, MultiPolygon):
        buffer_distance = step
        merged_polygon = multipolygon
        
        while merged_polygon.geom_type == 'MultiPolygon' and buffer_distance <= max_buffer_distance:
            buffered_polygons = [polygon.buffer(buffer_distance) for polygon in merged_polygon.geoms]
            merged_polygon = unary_union(buffered_polygons)
            buffer_distance += step
        
        if merged_polygon.geom_type == 'Polygon':
            # Remove the buffer to return to original size
            merged_polygon = merged_polygon.buffer(-buffer_distance + step)
        
        return merged_polygon
    
    return multipolygon




def get_polygon_coords_from_hull(hull):
    # handling both Polygon and MultiPolygon cases
    #print(f'{hull.geom_type=}')
    if isinstance(hull, MultiPolygon):
        largest_polygon = None
        max_area = 0

        # Iterate through all polygons to find the largest by area
        for polygon in hull.geoms:
            if polygon.area > max_area:
                largest_polygon = polygon
                max_area = polygon.area

        if largest_polygon:
            x, y = largest_polygon.exterior.xy
            return x, y
    elif hull.geom_type == 'Polygon':
        x, y = hull.exterior.xy
        return x, y
    else:
        
        print("The resulting geometry is neither a Polygon nor a MultiPolygon.")



def create_polygon_map_overlay(matrix, coords, hull, color="red", output_crs="EPSG:25833", folder='output/overlays/overlay', reduction_factor=5, search_id=0) -> Polygon:
    """
    Create a polygon map overlay from a matrix, coordinates, and a convex hull.

    Args:
        matrix (numpy.ndarray): The matrix representing the map.
        coords (tuple): The latitude and longitude coordinates of the map center.
        hull (scipy.spatial.ConvexHull): The convex hull of the map.
        color (str, optional): The color of the overlay. Defaults to "red".
        output_crs (str, optional): The coordinate reference system of the output. Defaults to "EPSG:25833".
        folder (str, optional): The folder to save the overlay file. Defaults to 'output/overlays/overlay'.
        reduction_factor (int, optional): The reduction factor for the map diameter. Defaults to 5.
        search_id (int, optional): The search ID. Defaults to 0.

    Returns:
        shapely.geometry.Polygon: The transformed polygon.

    """
    
    matrix_width, matrix_height = matrix.shape[0], matrix.shape[1]
    map_diameter = matrix_width * reduction_factor  # Meters
    distance_per_index = map_diameter / matrix_width  # Meters per index in the matrix
    lat, lng = coords
    center_x, center_y = transform_coordinates_to_utm(lat, lng)

    x, y = get_polygon_coords_from_hull(hull)
    hull_indices = list(zip(x, y))

    # Convert matrix indices to geographic coordinates with y-axis correction
    concave_hull_geo = []
    for x_idx, y_idx in hull_indices:
        # Calculate the meter position relative to the center
        x_meter = (x_idx - matrix_width / 2) * distance_per_index
        # Invert y-axis by subtracting y_idx from matrix_height before calculation
        y_meter = ((matrix_height - y_idx) - matrix_height / 2) * distance_per_index

        # Convert meter offsets to geographic coordinates
        x_geo, y_geo = center_x + x_meter, center_y + y_meter
        concave_hull_geo.append((x_geo, y_geo))

    # Create a polygon from the hull coordinates
    hull_polygon = Polygon(concave_hull_geo)
    # map polygon to coordinate reference system
    base_crs = 'EPSG:25833'
    #print(f'{hull_polygon=}')
    gdf = gpd.GeoDataFrame(index=[0], crs=base_crs, geometry=[hull_polygon])
    gdf.to_crs(output_crs, inplace=True)
    # save as GeoJSON
    gdf.to_file(f'{folder}id{search_id}_{color}_{lat}_{lng}_EPSG{output_crs[5:]}.geojson', driver='GeoJSON')
    print(f'Overlay saved to {folder}id{search_id}_{color}_{lat}_{lng}_EPSG{output_crs[5:]}.geojson')
    
    transformed_polygon = gdf.geometry.iloc[0]
    return transformed_polygon

            
def normalize_component(c):
    if c > 0:
        return 1
    elif c < 0:
        return -1
    return 0


def matrix_value_padding(matrix, value, padding=1, custom_offset=None):
    if custom_offset is None:
        padding_offset = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        padding_offset = custom_offset

    for _ in range(padding):
        rows, cols = np.where(matrix == value)
        
        for row, col in zip(rows, cols):
            for offset in padding_offset:
                new_row = row + offset[0]
                new_col = col + offset[1]
                try:
                    matrix[new_row, new_col] = value
                except:
                    pass
    return matrix


def rasterize_gdf(gdf, height, width, transform):
    raster = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(height, width),
            fill=0,
            transform=transform,
            all_touched=True
        )
    return raster


def normalize_array(array, cap):
    return array / cap


def downsample_rgb_image(image, factor):
    """
    Downsample an RGB image by selecting every 'factor'-th pixel from each dimension.
    Args:
    image (numpy array): 3-dimensional RGB image of shape (channels, height, width)
    factor (int): The downsampling factor for both height and width

    Returns:
    numpy array: The downsampled RGB image
    """
    # Downsampling the image by taking every 'factor'-th pixel along each spatial dimension
    downsampled_image = image[:, ::factor, ::factor]
    return downsampled_image


def downsample_2d_array(array, factor):
    """
    Downsample a 2D array by selecting every 'factor'-th element from each dimension.
    Args:
    array (numpy array): 2-dimensional array of shape (height, width)
    factor (int): The downsampling factor for both height and width

    Returns:
    numpy array: The downsampled 2D array
    """
    # Downsampling the array by taking every 'factor'-th element along each dimension
    downsampled_array = array[::factor, ::factor]
    return downsampled_array


def reduce_resolution(matrix, factor=5, method="mean"):
    if method == "mean":
        return matrix.reshape(matrix.shape[0]//factor, factor, matrix.shape[1]//factor, factor).mean(axis=1).mean(axis=2)
    elif method == "max":
        return matrix.reshape(matrix.shape[0]//factor, factor, matrix.shape[1]//factor, factor).max(axis=1).max(axis=2)
    elif method == "min":
        return matrix.reshape(matrix.shape[0]//factor, factor, matrix.shape[1]//factor, factor).mean(axis=1).min(axis=2)
    

def plot_branching_result(terrain_score_matrix, concave_hull_r, concave_hull_y, concave_hull_g, config, save=False):
    radius = terrain_score_matrix.shape[0] / 2
    plt.imshow(terrain_score_matrix, cmap='terrain', interpolation='nearest')
    plt.colorbar(label="Terreng: Vaskelig  ->  Lett")

    circle = Circle((radius, radius), (config.D25/config.REDUCTION_FACTOR), color="green", fill=False)   # search area circle
    plt.gca().add_patch(circle)

    circle = Circle((radius, radius), (config.D50/config.REDUCTION_FACTOR), color="yellow", fill=False)   # search area circle
    plt.gca().add_patch(circle)

    circle = Circle((radius, radius), (config.D75/config.REDUCTION_FACTOR), color="red", fill=False)   # search area circle
    plt.gca().add_patch(circle)

    if concave_hull_r:
        x_r, y_r = get_polygon_coords_from_hull(concave_hull_r)
        plt.fill(x_r, y_r, edgecolor='r',linewidth=3, fill=False)
    if concave_hull_y:
        x_y, y_y = get_polygon_coords_from_hull(concave_hull_y)
        plt.fill(x_y, y_y, edgecolor='y',linewidth=3, fill=False)
    if concave_hull_g:
        x_g, y_g = get_polygon_coords_from_hull(concave_hull_g)
        plt.fill(x_g, y_g, edgecolor='g',linewidth=3, fill=False)

    plt.title("Branching result plot")
    plt.axis('equal')

    if save:
        plt.savefig(f'{config.LOG_DIR}Simulation result.png')
        plt.close()
    else:
        plt.show()


def create_square_polygon(center_x, center_y, side_length):
    half_side = side_length / 2
    # Define the corners of the square
    corners = [
        (center_x - half_side, center_y - half_side),
        (center_x + half_side, center_y - half_side),
        (center_x + half_side, center_y + half_side),
        (center_x - half_side, center_y + half_side)
    ]
    return Polygon(corners)

def write_to_log_file(log_file, message):
    with open(log_file, 'a') as file:
        file.write(f'{message}')

def calculate_map_extension(max_distance, square_radius):
    """
    Calculates the map extension based on the maximum distance and square radius.

    Parameters:
    - max_distance (float): The maximum distance.
    - square_radius (float): The square radius.

    Returns:
    - map_extension (int): The calculated map extension.
    """
    
    map_square = 2*square_radius
    map_size = 2*max_distance
    print(f'{map_size=}')

    if map_size <= map_square:
        map_extension = 0
    elif map_size <= 3*map_square:
        map_extension = 1
    elif map_size <= 5*map_square:
        map_extension = 2
    elif map_size <= 7*map_square:
        map_extension = 3
    elif map_size <= 9*map_square:
        map_extension = 4
    elif map_size <= 11*map_square:
        map_extension = 5
    elif map_size <= 13*map_square:
        map_extension = 6
    else:
        map_extension = 7
    
    return map_extension