# https://github.com/dtlancaster/usgs-tiling/blob/master/usgs_tiling.py

import matplotlib.pyplot as plt
import numpy as py

"""
Takes the latitude and longitude as signed integers and constructs the appropriate file name for the TIF file.
"""


def construct_file_name(lat, lon):
    cardinal_1 = ''
    cardinal_2 = ''
    if lat < 0:
        cardinal_1 = 's'
    elif lat > 0:
        cardinal_1 = 'n'
    if lon < 0:
        cardinal_2 = 'w'
    elif lon > 0:
        cardinal_2 = 'e'
    abs_lat = abs(lat)
    abs_lon = abs(lon)
    file_name = 'USGS_NED_1_' + cardinal_1 + str(abs_lat) + cardinal_2 + '{0:0=3d}'.format(abs_lon) + '_IMG.tif'
    return file_name


"""
Takes the latitude and longitude as signed integers and loads the appropriate file. It then trims off the boundary of six pixels on all four sides.
"""


def load_trim_image(lat, lon):
    image = plt.imread(construct_file_name(lat, lon), format=None)
    image = image[6:(len(image) - 6), 6:(len(image) - 6)]
    return image


"""
Takes the northwest corner in degrees latitude and longitude and constructs a 2° by 2° elevation image. Each pixel represents a 1 x 1 arc-second, so the resulting image should be 7200 x 7200 (2 degrees = 7200 arc-seconds).
"""


def stitch_four(lat, lon):
    im1 = load_trim_image(lat, lon)
    im2 = load_trim_image(lat, lon + 1)
    im3 = load_trim_image(lat - 1, lon)
    im4 = load_trim_image(lat - 1, lon + 1)
    a = py.concatenate([im1, im3])
    b = py.concatenate([im2, im4])
    image = py.concatenate([a, b], axis=1)
    return image


"""
Takes the latitude, minimum longitude, and number of tiles and returns an image that combines tiles along a row of different longitudes.
"""


def get_row(lat, lon_min, num_tiles):
    rows = []
    for i in range(num_tiles):
        row = load_trim_image(lat, lon_min + i)
        rows.append(row)
        image = py.concatenate(rows, axis=1)
    return image


"""
Takes the northwest coordinate (maximum latitude, minimum longitude) and the number of tiles in each dimension (num_lat, num_lon) and constructs the image containing the entire range.
"""


def get_tile_grid(lat_max, lon_min, num_lat, num_lon):
    rows = []
    for i in range(num_lat):
        row = get_row(lat_max - i, lon_min, num_lon)
        rows.append(row)
        image = py.concatenate(rows)
    return image


"""
Get the integer coordinates of the northwest corner of the tile that contains this decimal (lat, lon) coordinate.
"""


def get_northwest(lat, lon):
    nw_lat = int(py.ceil(lat))
    nw_lon = int(py.floor(lon))
    return nw_lat, nw_lon


"""
Construct the tiled grid of TIF images that contains these northwest and southeast decimal coordinates. Each corner is a tuple (lat, lon).
"""


def get_tile_grid_decimal(northwest, southeast):
    begin_lat, begin_lon = get_northwest(northwest[0], northwest[1])
    end_lat, end_lon = get_northwest(southeast[0], southeast[1])
    num_lat = abs(int(py.fix(northwest[0])) - int(py.fix(southeast[0]))) + 1
    num_lon = abs(int(py.fix(northwest[1])) - int(py.fix(southeast[1]))) + 1
    image = get_tile_grid(begin_lat, begin_lon, num_lat, num_lon)
    return image
