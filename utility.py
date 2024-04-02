import pyproj
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiLineString, MultiPolygon
from shapely.ops import polygonize, unary_union
from scipy.spatial import Delaunay
import math
import random



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
        return (terrain * steepness) ** 2
        
    
def calc_travel_distance(matrix, energy, center, end_x, end_y, step_limit=9999):
    x0, y0 = center
    x1, y1 = end_x, end_y
    dx = abs(x1-x0)
    dy = abs(y1-y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    energy_used = 0
    steps = 0

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            energy_used += 1 / matrix[x][y]
            steps += 1
            err -= dy
            if err < 0:
                y += sy
                err += dx
                energy_used += 0.5 / matrix[x][y]
            x += sx
            if energy_used > energy or steps > step_limit:
                break
    else:
        err = dy / 2.0
        while y != y1:
            energy_used += 1 / matrix[x][y]
            steps += 1
            err -= dx
            if err < 0:
                x += sx
                err += dy
                energy_used += 0.5 / matrix[x][y]
            y += sy
            if energy_used > energy or steps > step_limit:
                break

    energy_left = energy - energy_used
    return (x, y), energy_left    # end point




def traverse(matrix, energy, start, target_direction, step_limit=9999):
    x0, y0 = start
    radians = math.radians(target_direction)
    
    # Calculate end points based on the step limit and direction
    dx = math.cos(radians) * step_limit
    dy = math.sin(radians) * step_limit
    # Theoretical end point considering no obstacles
    x1, y1 = x0 + dx, y0 + dy

    # Absolute values needed for Bresenham's algorithm
    dx = abs(dx)
    dy = abs(dy)
    
    # Determine the direction of steps (-1 for left/up, 1 for right/down)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    x, y = x0, y0
    energy_used = 0
    steps = 0
    err = dx - dy   # Error variable for Bresenham's line algorithm

    # Start traversal
    while True:
        # Check bounds to ensure the current position is within matrix
        if 0 <= x < matrix.shape[1] and 0 <= y < matrix.shape[0]:
            # Remember previous position before potential move
            prev_x, prev_y = x, y

            # If energy is depleted, step back to last valid position
            if energy_used > energy:
                x -= sx
                y -= sy
                break
        else:
            # If out of bounds, step back to last valid position
            x -= sx
            y -= sy
            break

        # If target is reached or steps exceed the limit, stop the traversal
        if x == int(x1) and y == int(y1) or steps > step_limit:
            break

        # Bresenham's algorithm for line drawing
        e2 = 2 * err
        if e2 >= -dy:
            err -= dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

        # Calculate energy cost based on the type of move and matrix values
        if (prev_x != x) and (prev_y != y):
            # Diagonal move, use √2 times the straight cost from the matrix
            energy_cost_diagonal = math.sqrt(2) / matrix[int(y)][int(x)]
            energy_used += energy_cost_diagonal
        else:
            # Straight move, cost is inverse of matrix value
            energy_cost_straight = 1 / matrix[int(y)][int(x)]
            energy_used += energy_cost_straight
        
        steps += 1
        
    energy_left = max(energy - energy_used, 0)
    return (int(x), int(y)), energy_left



def branching(matrix, x, y, angle, initial_energy, current_energy, sets):
    green, yellow, red = sets   # refrenced sets from main function
    if current_energy <= 0:
        red.add((x, y)) # red coords
        return

    # movement
    steps = 5   # steps to move 
    (new_x, new_y), new_energy = traverse(matrix, current_energy, (x, y), angle, steps)

    # save green and yellow coords if energy dips threshold
    if current_energy > initial_energy*0.66:
        if new_energy < initial_energy*0.66:
            green.add((new_x, new_y))
    elif current_energy > initial_energy*0.33:
        if new_energy < initial_energy*0.33:
            yellow.add((new_x, new_y))

    if new_energy > 0:
        # Branching conditions
        terrain_change = matrix[new_x, new_y] - matrix[x, y]
        if terrain_change < -0.05 and (new_x, new_y) not in red:
            for i in range(-2, 3, 1):
                branch_angle = angle + i * 45  # Calculate the new direction
                branch_angle %= 360  # Normalize angle
                branching(matrix, new_x, new_y, branch_angle, initial_energy, new_energy, sets)
        # Random branching
        elif random.randint(1,100) <= 5:    
            if (new_x, new_y) not in red and (new_x, new_y) not in yellow and (new_x, new_y) not in green:
                for i in range(-5, 6, 1):
                    branch_angle = angle + i * 10  # Calculate the new direction
                    branch_angle %= 360  # Normalize angle
                    branching(matrix, new_x, new_y, branch_angle, initial_energy, new_energy, sets)
        else:
            # Continue in the same direction
            branching(matrix, new_x, new_y, angle, initial_energy, new_energy, sets)
        
    else:
        # Energy depleted, stop recursion and save end point
        red.add((new_x, new_y)) # red coords
        




def plot_array(array, cmap="terrain", label=""):
    plt.imshow(array, cmap=cmap)
    plt.colorbar(label=label)
    plt.axis('off')
    plt.show()


def compute_concave_hull_from_points(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.
    @param points: Iterable container of points.
    @param alpha: Alpha value to influence the gooeyness of the border. Smaller numbers
                  don't fall inward as much as larger numbers. Too large, and you lose everything!
    """
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
    return concave_hull

def get_polygon_from_hull(hull):
    # handling both Polygon and MultiPolygon cases
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