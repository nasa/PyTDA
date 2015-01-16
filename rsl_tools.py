import numpy as np

"""
Version 1.0
Last Updated 12/09/2014

"""

def rsl_get_groundr_and_h(slant_r, elev):

    """
Author: Timothy Lang (timothy.j.lang@nasa.gov)
Given slant range and elevation, return ground range and height.

Inputs
  slant_r  slant range (km)
  elev     elevation (deg)

Outputs
  gr    ground range (km)
  h     height above radar (km)

This Python function is adapted from the Radar Software Library routine
RSL_get_groundr_and_h, written by John Merritt and Dennis Flanigan

    """

    Re = 4.0/3.0 * 6371.1  # Effective earth radius in km.
    h = np.sqrt(Re**2 + slant_r**2 - 2.0 * Re * slant_r * np.cos((elev + 90.0) * np.pi / 180.0))
    gr = Re * np.arccos( ( Re**2 + h**2 - slant_r**2) / (2.0 * Re * h))
    h = h - Re
    return gr, h

###################################################

def rsl_get_slantr_and_elev(gr, h):

    """
Author: Timothy Lang (timothy.j.lang@nasa.gov)
Given ground range and height, return slant range and elevation.

Inputs
  gr  ground range (km)
  h   height (km)

Outputs
  slantr    slant range
  elev      elevation in degrees

This Python function is adapted from the Radar Software Library routine
RSL_get_slantr_and_elev, written by John Merritt and Dennis Flanigan

    """

    Re = 4.0/3.0 * 6371.1  # Effective earth radius in km.
    rh = h + Re
    slantrsq = Re**2 + rh**2 - (2 * Re * rh * np.cos(gr/Re))
    slantr = np.sqrt(slantrsq)
    elev = np.arccos((Re**2 + slantrsq - rh**2)/(2 * Re * slantr))
    elev = elev * 180.0/np.pi
    elev = elev - 90.0
    return slantr, elev


