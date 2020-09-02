import os

import cv2
import h5py
import numpy as np
import tifffile as tiff
import datetime


def read_volume(file, type='tif3d', key=None, start=0, stop=-1, dtype='uint8'):
    """
    Reads a volume file/directory and returns the data in it as a numpy array

    :param file: path to the data
    :param type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param key: key to the data (only necessary for hdf5 files)
    :param start: first slice to read (only necessary for sequences)
    :param stop: last slice (exclusive) to read (-1 means it will read up to the final slice, only necessary for sequences)
    :param dtype: data type of the numpy array
    :return: numpy array containing the data
    """

    if type == 'tif2d' or type == 'tif3d' or type == 'tif2D' or type == 'tif3D':
        volume = read_tif(file, dtype=dtype)
    elif type == 'tifseq':
        volume = read_tifseq(file, start=start, stop=stop, dtype=dtype)
    elif type == 'hdf5':
        volume = read_hdf5(file, key=key, dtype=dtype)
    elif type == 'png':
        volume = read_png(file, dtype=dtype)
    elif type == 'pngseq':
        volume = read_pngseq(file, start=start, stop=stop, dtype=dtype)
    else:
        volume = None

    return volume


def read_tif(file, dtype='uint8'):
    """
    Reads tif formatted file and returns the data in it as a numpy array

    :param file: path to the tif file
    :param dtype: data type of the numpy array
    :return: numpy array containing the data
    """

    data = tiff.imread(file).astype(dtype)

    return data


def read_tifseq(dir, dtype='uint8', start=0, stop=-1):
    """
    Read a sequence of 2D TIF files

    :param dir: directory that contains the files
    :param dtype: data type of the output
    :param start: first slice to read
    :param stop: last slice (exclusive) to read (-1 means it will read up to the final slice)
    """

    files = os.listdir(dir)
    files.sort()
    sz = tiff.imread(os.path.join(dir, files[0])).shape
    if stop < 0:
        stop = len(files)
    data = np.zeros((stop - start, *sz), dtype=dtype)
    for i in range(start, stop):
        data[i-start] = tiff.imread(os.path.join(dir, files[i]))

    return data


def read_hdf5(file, dtype='uint8', key=None):
    """
    Reads an HDF5 file as a numpy array

    :param file: path to the hdf5 file
    :param dtype: data type of the numpy array
    :param key: key in the hfd5 file that provides access to the data
    :return: numpy array containing the data
    """
    f = h5py.File(file, 'r')
    data = np.array(f.get(key), dtype=dtype)
    f.close()

    return data


def read_png(file, dtype='uint8'):
    """
    Read a 2D PNG file

    :param file: file to be read
    :param dtype: data type of the output
    """

    data = cv2.imread(file, cv2.IMREAD_ANYDEPTH).astype(dtype)

    return data


def read_jpg(file, dtype='uint8'):
    """
    Read a 2D JPG file

    :param file: file to be read
    :param dtype: data type of the output
    """

    data = cv2.imread(file, cv2.IMREAD_ANYDEPTH).astype(dtype)

    return data


def read_pngseq(dir, dtype='uint8', start=0, stop=-1):
    """
    Read a sequence of 2D PNG files

    :param dir: directory that contains the files
    :param dtype: data type of the output
    :param start: first slice to read
    :param stop: last slice (exclusive) to read (-1 means it will read up to the final slice)
    """

    files = os.listdir(dir)
    files.sort()
    sz = cv2.imread(os.path.join(dir, files[0]), cv2.IMREAD_ANYDEPTH).shape
    if stop < 0:
        stop = len(files)
    data = np.zeros((stop - start, sz[0], sz[1]), dtype=dtype)
    for i in range(start, stop):
        data[i-start] = cv2.imread(os.path.join(dir, files[i]), cv2.IMREAD_ANYDEPTH).astype(dtype)

    return data


def write_volume(data, file, type='tif3d', index_inc=0, start=0, stop=-1, dtype='uint8', K=4):
    """
    Writes a numpy array to a volume file/directory

    :param data: 2D/3D numpy array
    :param file: path to the data
    :param type: type of the volume file (tif2d, tif3d, tifseq, png or pngseq)
    :param index_inc: increment for the index filename (only necessary for sequences)
    :param start: first slice to write (only necessary for sequences)
    :param stop: last slice (exclusive) to write (-1 means it will write up to the final slice, only necessary for sequences)
    :param dtype: data type of the output
    :param K: length of the string index (optional, only for sequences)
    """

    if type == 'tif2d' or type == 'tif3d':
        write_tif(data, file, dtype=dtype)
    elif type == 'tifseq':
        write_tifseq(data, file, index_inc=index_inc, start=start, stop=stop, dtype=dtype, K=K)
    elif type == 'png':
        write_png(data, file, dtype=dtype)
    elif type == 'pngseq':
        write_pngseq(data, file, index_inc=index_inc, start=start, stop=stop, dtype=dtype, K=K)


def write_tif(x, file, dtype='uint8'):
    """
    Write a 2D/3D data set as a TIF file

    :param x: 2D/3D data array
    :param file: directory to write the data to
    :param dtype: data type of the output
    """

    tiff.imsave(file, x.astype(dtype))


def write_png(x, file, dtype='uint8'):
    """
    Write a 2D data set to a PNG file

    :param x: 3D data array
    :param file: directory to write the data to
    :param dtype: data type of the output
    """

    cv2.imwrite(file, x.astype(dtype), [cv2.IMWRITE_PNG_COMPRESSION, 9])


def write_jpg(x, file, quality=100, dtype='uint8'):
    """
    Write a 2D data set to a JPEG file

    :param x: 3D data array
    :param file: directory to write the data to
    :param quality: quality of the JPEG compression (0-100)
    :param dtype: data type of the output
    """

    cv2.imwrite(file, x.astype(dtype), [cv2.IMWRITE_JPEG_QUALITY, quality])


def write_tifseq(x, dir, prefix='', index_inc=0, start=0, stop=-1, dtype='uint8', K=4):
    """
    Write a 3/4D data set to a directory, slice by slice, as TIF files

    :param x: 3/4D data array
    :param dir: directory to write the data to
    :param prefix: prefix of the separate files
    :param index_inc: increment for the index filename (only necessary for sequences)
    :param start: first slice to write
    :param stop: last slice (exclusive) to write (-1 means it will read up to the final slice)
    :param dtype: data type of the output
    :param K: number of digits for the index
    """

    if not os.path.exists(dir):
        os.mkdir(dir)
    if stop < 0:
        stop = x.shape[0]
    for i in range(start, stop):
        i_str = _num2str(index_inc + i, K=K)
        tiff.imsave(dir + '/' + prefix + i_str + '.tif', (x[i, ...]).astype(dtype))


def write_pngseq(x, dir, prefix='', index_inc=0, start=0, stop=-1, dtype='uint8', K=4):
    """
    Write a 3D data set to a directory, slice by slice, as PNG files

    :param x: 3D data array
    :param dir: directory to write the data to
    :param prefix: prefix of the separate files
    :param index_inc: increment for the index filename (only necessary for sequences)
    :param start: first slice to write
    :param stop: last slice (exclusive) to write (-1 means it will read up to the final slice)
    :param dtype: data type of the output
    :param K: number of digits for the index
    """

    if not os.path.exists(dir):
        os.mkdir(dir)
    if stop < 0:
        stop = x.shape[0]
    for i in range(start, stop):
        i_str = _num2str(index_inc + i, K=K)
        cv2.imwrite(dir + '/' + prefix + i_str + '.png', (x[i, :, :]).astype(dtype),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])


def mkdir(filename):
    if not os.path.exists(filename):
        os.mkdir(filename)


def print_frm(s, time=True, flush=True):
    if time:
        print('[%s] %s' % (datetime.datetime.now(), s), flush=flush)
    else:
        print(s, flush=flush)


def _num2str(n, K=4):
    n_str = str(n)
    for k in range(0, K - len(n_str)):
        n_str = '0' + n_str
    return n_str
