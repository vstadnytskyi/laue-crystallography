#!/usr/bin/env python3
"""
Authors: Valentyn Stadnytskyi
Date created: 23 Oct 2019
Date last modified: 25 Oct 2019
Python Version: 3

Description:
------------
The Template Server hosts three PVs: CMD(command PV), ACK(acknowledgement PV) and values PV.
"""



from numpy import array, zeros

pixel_size = 1 #in microns
plane = zeros((1024,1024))

def distance(x,y):
    pass

def tau(length, diameter, viscosity, compressibility):
    """
    convert the length of a plug to volume in 250 um capillary.

    Parameters
    ----------
    length : float
        cylinder length
    diameter : float
        cylinder diameter

    Returns
    -------
    tau : float
        tau constant in seconds

    Examples
    --------
    the example of usage

    >>> droplet_volume(250,250)
    """
    tau = 32*compressibility*viscosity*(length/diameter)**2
    return tau


def droplet_volume(length = 100, diameter = 250):
    """
    convert the length of a plug to volume in 250 um capillary.

    Parameters
    ----------
    l : float
        droplet or plug length
    d : float
        channel diameter

    Returns
    -------
    v : float
        volume of the object

    Examples
    --------
    the example of usage

    >>> droplet_volume(250,250)
    """

def flow_cylinder(diameter = 250, length = 1, viscosity = 1, pressure = 0.1, output_units = 'nL'):
    """
    calculates flow rate via a cylindrical capillary with a given diameter, length and fluid viscosity under a given pressure drop. Also takes string value to define output units

    Parameters
    ----------
    length : float
        length of a cylinder in meters
    diameter : float
        diameter of round cross-section in micrometer
    viscosity : float
        viscosity in cP or mPa*s
    pressure : float
        pressure difference in atmospheres
    output_units
        output units for the flow. Supports: nL/s, uL/s. Default: nL/s

    Returns
    -------
    flow : float
        flow of fluid in nL/s

    Examples
    --------
    the example of usage.
    D=50um,L = 1, eta = 1, P = 2 -> 31.086 nL/s
    D=10oum,L = 0.7, eta = 23, P = 2 -> 30.893 nL/s


    >>> flow_cylinder(diameter = 50, length = 1, viscosity = 1.0, pressure = 2.0)
    >>> 31.086120666502506
    """
    from numpy import pi
    #first convert to SI units
    diameter = diameter*10**-6
    length = length
    viscosity = viscosity*10**-3 # 1 cP = 1mPa*s
    pressure = pressure*101325 # 1 atm = 101325 Pascal
    flow = (pi * pressure*diameter**4) / (128*viscosity*length)
    if output_units == 'pL':
        output_factor = 10**15 # 1 m^3 -> 1e3 L -> 1e12 nL
    elif output_units == 'nL':
        output_factor = 10**12 # 1 m^3 -> 1e3 L -> 1e12 nL
    elif output_units == 'uL':
        output_factor = 10**9 # 1 m^3 -> 1e3 L -> 1e12 nL
    elif output_units == 'mL':
        output_factor = 10**6 # 1 m^3 -> 1e3 L -> 1e12 nL
    elif output_units == 'L':
        output_factor = 10**3 # 1 m^3 -> 1e3 L -> 1e12 nL
    else:
        output_factor = 10**12
    return flow*output_factor

def pressure_cylinder(diameter = 250, length = 1, viscosity = 1, flow = 0.1, flow_units = 'nL/s', output_units = 'atm'):
    """
    calculates pressure drop across a cylindrical capillary with a given diameter, length and fluid viscosity with given flow rate. Also takes string value to define output units

    Parameters
    ----------
    length : float
        length of a cylinder in meters
    diameter : float
        diameter of round cross-section in micrometer
    viscosity : float
        viscosity in cP or mPa*s
    flow : float
        pressure difference in atmospheres
    flow_units : string
        flow units (pL/s, uL/s, ul/s, mL/s, L/s, m^3/s)
    output_units : string
        output units for the flow. Supports: atm, Pa. Default: atm

    Returns
    -------
    pressure : float
        pressure drop across the capillary

    Examples
    --------
    the example of usage.
    D=50um,L = 1, eta = 1, P = 2 -> 31.086 nL/s
    D=10oum,L = 0.7, eta = 23, P = 2 -> 30.893 nL/s


    >>> pressure_cylinder(diameter = 50, length = 1, viscosity = 1.0, flow = 31.086, flow_units = 'nL/s')
    >>> 1.9999922366316594
    """
    from numpy import pi
    #first convert to SI units
    diameter = diameter*10**-6
    length = length
    viscosity = viscosity*10**-3 # 1 cP = 1mPa*s
    if flow_units == 'nL/s':
        flow_coeff = 10**-12
    elif flow_units == 'uL/s':
        flow_coeff = 10**-9
    elif flow_units == 'mL/s':
        flow_coeff = 10**-6
    flow = flow*flow_coeff #
    pressure = ((128*viscosity*length)/(pi *diameter**4))*flow
    if output_units == 'atm':
        output_factor = 1/101325 # 1 m^3 -> 1e3 L -> 1e12 nL
    elif output_units == 'Pa':
        output_factor = 1 # 1 m^3 -> 1e3 L -> 1e12 nL
    else:
        output_factor = 1/101325 # 1 m^3 -> 1e3 L -> 1e12 nL
    return pressure*output_factor

def find_circles(image):
    """
    takes and input image and find circles in it.

    returns radius and center coordinates of circles
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from skimage import data, color
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.feature import canny
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte


    # Load picture and detect edges
    image = img_as_ubyte(image)
    edges = canny(image, sigma=2, low_threshold=10, high_threshold=20)


    # Detect two radii
    hough_radii = np.arange(40, 280, 5)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=7)

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
        image[circy, circx] = (220, 20, 20)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()

def get_test_images(i = 0):
    from ubcs_auxiliary.save_load_object import load_from_file
    filename = './laue_crystallography/tests/images/holes_example1.imgpkl'
    img = load_from_file(filename)
    return img

if __name__ == '__main__':
    # The Record Name is specified by prefix
    prefix = 'microfludics'
    from pdb import pm
    from tempfile import gettempdir
    import logging
    logging.basicConfig(filename=gettempdir()+'/{}.log'.format(prefix),
                        level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")

    from tempfile import gettempdir
