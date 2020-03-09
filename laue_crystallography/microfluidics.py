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

def flow(diameter = 250, length = 1, viscosity = 1, pressure = 0.1):
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
if __name__ == '__main__':
    # The Record Name is specified by prefix
    prefix = 'microfludics'
    from pdb import pm
    from tempfile import gettempdir
    import logging
    logging.basicConfig(filename=gettempdir()+'/{}.log'.format(prefix),
                        level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")

    from tempfile import gettempdir
