"""
Python Turbulence Detection Algorithm (PyTDA)
Version 1.1.2
Last Updated 07/31/2019


Major References
----------------
Bohne, A. R. (1982). Radar detection of turbulence in precipitation
    environments. Journal of the Atmospheric Sciences, 39(8), 1819-1837.
Doviak, R. J., and D. S. Zrnic, 1993: Doppler Radar and Weather Observations,
    Academic Press, 562 pp.
Labitt, M. (1981). Coordinated radar and aircraft observations of turbulence
    (No. ATC-108). Federal Aviation Administration, Systems Research and
    Development Service.
Williams, J. K., L. B. Cornman, J. Yee, S. G. Carson, G. Blackburn, and
    J. Craig, 2006: NEXRAD detection of hazardous turbulence. 44th AIAA
    Aerospace Sciences Meeting and Exhibit, Reno, NV.


Author
------
Timothy James Lang
timothy.j.lang@nasa.gov
(256) 961-7861


Overview
--------
This software will estimate the cubic root of eddy dissipation rate, given
input radar data containing reflectivity and spectrum width. Can be done
on an individual sweep basis or by processing a full volume at once. If
the latter, a new turbulence field is created within the Py-ART radar object.
Based on the NCAR Turbulence Detection Algorithm (NTDA). For 2010 and older
NEXRAD data (V06 and earlier), recommend running on CF Radials produced from
native Level 2 files via Radx due to conflicts between Py-ART and older NEXRAD
data models.


Change Log
----------
Version 1.1.2 Major Changes (07/31/2019):
1. Fixed a bug in the RHI code that was preventing processing when
   use_ntda=False.

Version 1.1.1 Major Changes (11/27/2015):
1. Added common sub-module with old radar_coords_to_cart function that was
   in old version of Py-ART. Recent upgrade of Py-ART removed this function,
   breaking PyTDA in the process.

Version 1.1 Major Changes (10/29/2015):
1. Fixed more issues for when radar object fields lack masks or fill values.
2. Enabled RHI functionality

Version 1.0 Major Changes (08/28/2015):
1. Fixed issues for when radar object fields lack masks or fill values.
2. Fixed failure when radar.sweep_number attribute is not sequential

Version 0.9 Major Changes (08/03/2015):
1. Made compliant with Python 3.

Version 0.8 Major Changes (07/02/2015):
1. Made all code pep8 compliant.

Version 0.7 Major Changes (03/16/2015):
1. Minor edits to improve documentation and reduce number of local variables.

Version 0.6 Major Changes (11/26/2014):
1. Changed from NTDA's lookup table to basic equations for relating spectrum
   width to EDR. Now can account for radars with other beamwidths and
   gate spacings than NEXRAD.
2. Added use_ntda flag to turn on/off NTDA-based filtering (i.e., can turn off
   to straight up convert SW to EDR).
3. Performance improvements leading to significant code speedup.
4. Removed variables and imports related to NTDA lookup tables.
5. Changed name of different_sweeps flag to split_cut for improved clarity.
6. Changed atan2_cython function's name to atan2c_longitude to better indicate
   its specialized nature. Added more generic atan2c function to
   pytda_cython_tools, although this is not currently used.

Version 0.5 Major Changes:
1. Fixed a bug that prevented actual filtering of DZ/SW fields before
   turbulence calculations in certain radar volumes. Performance improved
   considerably!
2. Fixed bug that set turbulence to 0 where it was never calculated to
   begin with. Should instead be the bad data fill value for spectrum width.
3. Fixed bug that caused a crash when running calc_turb_vol() with
   different_sweeps keyword set to True.

Version 0.4 Major Changes:
1. Refactoring to reduce number of local variables in calc_turb_sweep() proper.
2. Added calc_turb_vol(), which leverages calc_turb_sweep() to process and
   entire volume at once, and add the turbulence field to the Py-ART radar
   object.
3. Added add_turbulence_field() to create a Py-ART radar field for turbulence

Version 0.3 Major Changes:
1. Refactoring of calc_turb_sweep() to drastically speed up processing.

Version 0.2 Functionality:
1. calc_turb_sweep() - Input Py-ART radar object, receive back turbulence on
   the specified sweep plus longitude and latitude in that coordinate system.

"""

from __future__ import print_function
import numpy as np
from sklearn.neighbors import BallTree
from scipy.special import gamma as gamma
from scipy.special import hyp2f1 as hypergeometric_gaussian
import time
import pyart
from .common import radar_coords_to_cart
from .rsl_tools import rsl_get_groundr_and_h
from .pytda_cython_tools import calc_cswv_cython, atan2c_longitude

VERSION = '1.1.1'

# sw_* as prefix = related to sweep
# *_sw as suffix = related to spectrum width
re = 6371.1  # km
MAX_INT = [-300.0, 300.0]  # km
VARIANCE_RADIUS_SW = 1.0  # km
DEFAULT_RADIUS = 2.0  # km
DEFAULT_DZ = 'reflectivity'
DEFAULT_SW = 'spectrum_width'
DEFAULT_TURB = 'turbulence'
DEFAULT_BEAMWIDTH = 0.96  # deg
DEFAULT_GATE_SPACING = 0.25  # km
RNG_MULT = 1000.0  # m per km
SPLIT_CUT_MAX = 2  # maximum number of split cut sweeps
RRV_SCALING_FACTOR = (8.0 * np.log(4.0))**0.5  # From Bohne (1982)
KOLMOGOROV_CONSTANT = 1.6
CONSTANT = KOLMOGOROV_CONSTANT * gamma(2.0/3.0)
BAD_DATA_VAL = -32768

# TO DO: Class definition here to simplify calc_turb_sweep() code?


def calc_turb_sweep(radar, sweep_number, radius=DEFAULT_RADIUS,
                    split_cut=False,
                    xran=MAX_INT, yran=MAX_INT, verbose=False,
                    name_dz=DEFAULT_DZ, name_sw=DEFAULT_SW,
                    use_ntda=True, beamwidth=DEFAULT_BEAMWIDTH,
                    gate_spacing=DEFAULT_GATE_SPACING):
    """
    Provide a Py-ART radar object containing reflectivity and spectrum width
    variables as an argument, along with the sweep number and any necessary
    changes to the keywords, and receive back turbulence for the same sweep
    as spectrum width (along with longitude and latitude on the same coordinate
    system).
    radar = Py-ART radar object
    sweep_number = Can be as low as 0, as high as # of sweeps minus 1
    radius = radius of influence (km)
    split_cut = Set to True if using NEXRAD or similar radar that has two
                separate low-level sweeps at the same tilt angle for DZ & SW
    verbose = Set to True to get more information about calculation progress.
    xran = [Min X from radar, Max X from radar], subsectioning improves
           performance
    yran = [Min Y from radar, Max Y from radar], subsectioning improves
           performance
    name_dz = Name of reflectivity field, used by Py-ART to access field
    name_sw = Name of spectrum width field, used by Py-ART to access field
    use_ntda = Flag to use the spatial averaging and weighting employed by NTDA
    beamwidth = Beamwidth of radar in degrees
    gate_spacing = Gate spacing of radar in km

    """

    if verbose:
        overall_time = time.time()
        begin_time = time.time()

    sweep_dz = get_sweep_data(radar, name_dz, sweep_number)
    try:
        fill_val_sw = radar.fields[name_sw]['_FillValue']
    except KeyError:
        fill_val_sw = BAD_DATA_VAL
    try:
        fill_val_dz = radar.fields[name_dz]['_FillValue']
    except KeyError:
        fill_val_dz = BAD_DATA_VAL
    sweep_sw, sweep_az_sw, sweep_elev_sw, dz_sw = \
        _retrieve_sweep_fields(radar, name_sw, name_dz, sweep_number,
                               sweep_dz, split_cut)

    # Radar location needed to get lat/lon
    # of every gate for distance calculations
    klat, klon, klatr, klonr = get_radar_latlon_plus_radians(radar)

    # Initialize information on sweep geometry
    sw_lonr = 0.0 * sweep_sw
    sw_azr_1d = np.deg2rad(sweep_az_sw)
    sw_sr_1d = radar.range['data'][:] / RNG_MULT
    result = np.meshgrid(sw_sr_1d, sweep_az_sw)
    sw_azr_2d = np.deg2rad(result[1])
    sw_sr_2d = result[0]
    result = np.meshgrid(sw_sr_1d, sweep_elev_sw)
    sw_el = result[1]
    sweep_size = radar.sweep_end_ray_index['data'][sweep_number] + 1 - \
        radar.sweep_start_ray_index['data'][sweep_number]
    sw_gr, sw_ht = rsl_get_groundr_and_h(sw_sr_2d, sw_el)
    sw_latr = np.arcsin((np.sin(klatr) * np.cos(sw_gr/re)) +
                        (np.cos(klatr) * np.sin(sw_gr/re) * np.cos(sw_azr_2d)))

    if verbose:
        print(time.time() - begin_time,
              'seconds to complete all preliminary processing')
        begin_time = time.time()

    # Get longitude at every gate
    for j in np.arange(sweep_size):
        for i in np.arange(radar.ngates):
            sw_lonr[j, i] = klonr +\
                atan2c_longitude(sw_azr_1d[j], sw_gr[j, i],
                                 klatr, sw_latr[j, i])
    sw_lat = np.rad2deg(sw_latr)
    sw_lon = np.rad2deg(sw_lonr)

    # Determine NTDA interest fields
    csnr_sw = _calc_csnr_for_every_gate(dz_sw, sw_sr_2d)
    crng_sw = _calc_crng_for_every_gate(sw_sr_2d)
    czh_sw = _calc_czh_for_every_gate(dz_sw, sw_ht)
    cpr = 1.0  # Not calculating Cpr, user must remove second trip manually.

    if verbose:
        print(time.time() - begin_time,
              'seconds to compute longitudes and precompute Csnr, Crng, Czh')
        begin_time = time.time()

    xx, yy = calc_cartesian_coords_radians(sw_lonr, sw_latr, klonr, klatr)

    # Flatten all arrays to avoid performance buzzkill of nested loops.
    # Then reduce the data using masks to make the job manageable.
    sweep_sw = sweep_sw.ravel()
    dz_sw = dz_sw.ravel()
    condition = np.logical_and(dz_sw != fill_val_dz, sweep_sw != fill_val_sw)
    sweep_sw = sweep_sw[condition]
    dz_sw = dz_sw[condition]

    # Brief detour to get eps and check to see if that's all user wants.
    # Placing this code here with reduced sweep_sw improves performance.
    sr_1d = flatten_and_reduce_data_array(sw_sr_2d, condition)
    cond = gate_spacing >= sr_1d * np.deg2rad(beamwidth)
    eps_sw = edr_long_range(sweep_sw, sr_1d, beamwidth, gate_spacing)
    eps_sw[cond] = edr_short_range(sweep_sw[cond], sr_1d[cond],
                                   beamwidth, gate_spacing)
    if not use_ntda:
        turb_radar_f = 1.0 * eps_sw
        turb_radar = sw_lat.flatten() * 0.0 + fill_val_sw
        turb_radar[condition] = turb_radar_f
        eps_sw = np.reshape(turb_radar, (len(sweep_az_sw), radar.ngates))
        if verbose:
            print(time.time() - overall_time,
                  'seconds to process radar sweep')
        return eps_sw, sw_lat, sw_lon

    # Provided NTDA is wanted, continue flattening/reducing arrays
    xx = flatten_and_reduce_data_array(xx, condition)
    yy = flatten_and_reduce_data_array(yy, condition)
    csnr_sw = flatten_and_reduce_data_array(csnr_sw, condition)
    crng_sw = flatten_and_reduce_data_array(crng_sw, condition)
    czh_sw = flatten_and_reduce_data_array(czh_sw, condition)
    turb_radar_f = 0.0 * sweep_sw + fill_val_sw

    # Find the distance to every other good gate
    ind, ind_sw = _calc_tree(xx, yy, radius)
    cswv_sw = _calc_cswv_for_every_gate(xx, sweep_sw, ind_sw)
    if verbose:
        print(time.time() - begin_time,
              'seconds to get eps, reduce data,',
              'compute BallTree, and get Cswv')
        begin_time = time.time()

    # Loop thru data and do NTDA filtering
    for i in np.arange(len(xx)):
        if verbose:
            if i % 50000 == 0:
                print('i =', i, 'of', len(xx) - 1,
                      time.time() - begin_time, 'seconds elapsed during loop')
        if xran[0] < xx[i] < xran[1] and yran[0] < yy[i] < yran[1]:
            # Broadcating employed to minimize the amount of looping
            eps = eps_sw[ind[i]]**2
            csnr = csnr_sw[ind[i]]**0.6667
            crng = crng_sw[ind[i]]
            czh = czh_sw[ind[i]]
            cswv = cswv_sw[ind[i]]
            # Begin NTDA-specific calculation
            tot = csnr * cpr * cswv * czh * crng
            num = tot * eps
            tot = np.sum(tot)
            num = np.sum(num)
            if tot > 0:
                turb_radar_f[i] = np.sqrt(num/tot)

    # Restore turbulence to a full 2-D sweep array and return along w/ lat/lon.
    turb_radar = sw_lat.flatten() * 0.0 + fill_val_sw
    turb_radar[condition] = turb_radar_f
    turb_radar = np.reshape(turb_radar, (len(sweep_az_sw), radar.ngates))
    if verbose:
        print(time.time() - overall_time, 'seconds to process radar sweep')
    return turb_radar, sw_lat, sw_lon

#######################################


def calc_turb_vol(radar, radius=DEFAULT_RADIUS, split_cut=False,
                  xran=MAX_INT, yran=MAX_INT, verbose=False,
                  name_dz=DEFAULT_DZ, name_sw=DEFAULT_SW,
                  turb_name=DEFAULT_TURB, max_split_cut=SPLIT_CUT_MAX,
                  use_ntda=True, beamwidth=DEFAULT_BEAMWIDTH,
                  gate_spacing=DEFAULT_GATE_SPACING):
    """
    Leverages calc_turb_sweep() to process an entire radar volume for
    turbulence. Has ability to account for split-cut sweeps in a volume
    (i.e., DZ & SW on different, mismatched sweeps).
    radar = Py-ART radar object
    radius = Search radius for calculating EDR
    split_cut = Set to True for split-cut volumes
    xran = Spatial range in X to consider
    yran = Spatial range in Y to consider
    verbose = Set to True to get more information on calculation status
    name_dz = Name of reflectivity field
    name_sw = Name of spectrum width field
    turb_name = Name for created turbulence field
    max_split_cut = Total number of tilts that are affected by split cuts
    use_ntda = Flag to use the spatial averaging and weighting employed by NTDA
    beamwidth = Beamwidth of radar in degrees
    gate_spacing = Gate spacing of radar in km
    """

    if verbose:
        vol_time = time.time()
    fill_value, turbulence = _initialize_turb_field(radar, name_sw)

    # Commented section fails if sweep_number not [0, 1, 2, 3 ...]
    # index = np.min(radar.sweep_number['data'])
    # while index <= np.max(radar.sweep_number['data']):
    index = 0
    while index < radar.nsweeps:
        if verbose:
            print('Sweep number:', index)
        if split_cut and index < max_split_cut:
            ind_adj = index + 1
            dsw = True
        else:
            ind_adj = index
            dsw = False
        try:
            sweep_range = [radar.sweep_start_ray_index['data'][ind_adj],
                           radar.sweep_end_ray_index['data'][ind_adj]+1]
            turbulence[sweep_range[0]:sweep_range[1]], glat, glon = \
                calc_turb_sweep(radar, index, radius=radius, split_cut=dsw,
                                verbose=verbose, xran=xran, yran=yran,
                                name_dz=name_dz, name_sw=name_sw,
                                use_ntda=use_ntda, beamwidth=beamwidth,
                                gate_spacing=gate_spacing)
        except IndexError:
            print('Ran out of sweeps')
        finally:
            index += 1
            if split_cut and index < max_split_cut:
                index += 1

    turbulence = _finalize_turb_field(radar, turbulence, name_dz, name_sw)
    add_turbulence_field(radar, turbulence, turb_name)
    if verbose:
        print((time.time()-vol_time)/60.0, 'minutes to process volume')

###################################


def calc_turb_rhi(radar, radius=1.0, verbose=False,
                  name_dz=DEFAULT_DZ, name_sw=DEFAULT_SW,
                  turb_name=DEFAULT_TURB, max_split_cut=SPLIT_CUT_MAX,
                  use_ntda=True, beamwidth=None,
                  gate_spacing=DEFAULT_GATE_SPACING):
    """
    Processes an entire RHI radar volume for turbulence.
    radar = Py-ART radar object
    radius = Search radius for calculating EDR
    verbose = Set to True to get more information on calculation status
    name_dz = Name of reflectivity field
    name_sw = Name of spectrum width field
    turb_name = Name for created turbulence field
    use_ntda = Flag to use the spatial averaging and weighting employed by NTDA
    beamwidth = Beamwidth of radar in degrees
    gate_spacing = Gate spacing of radar in km
    """
    if verbose:
        vol_time = time.time()

    fill_value, turbulence = _initialize_turb_field(radar, name_sw)
    if beamwidth is None:
        beamwidth = radar.instrument_parameters['radar_beam_width_v']['data']

    index = 0
    while index < radar.nsweeps:
        if verbose:
            print('Sweep number:', index)
        ind_adj = index
        try:
            sweep_range = [radar.sweep_start_ray_index['data'][ind_adj],
                           radar.sweep_end_ray_index['data'][ind_adj]+1]
            turbulence[sweep_range[0]:sweep_range[1]] = \
                _calc_turb_rhi_sweep(
                    radar, index, radius=radius, verbose=verbose,
                    name_dz=name_dz, name_sw=name_sw,
                    use_ntda=use_ntda, beamwidth=beamwidth,
                    gate_spacing=gate_spacing)
        except IndexError:
            print('Ran out of sweeps')
        finally:
            index += 1

    turbulence = _finalize_turb_field(radar, turbulence, name_dz, name_sw)
    add_turbulence_field(radar, turbulence, turb_name)
    if verbose:
        print((time.time()-vol_time)/60.0, 'minutes to process volume')

###################################


def edr_long_range(sw, rng, theta, gs):
    """
    For gate spacing < range * beamwidth
    sw (spectrum width) in m/s,
    rng (range) in km,
    theta (beamwidth) in deg,
    gs (gate spacing) in km

    """
    beta = gs * RNG_MULT / RRV_SCALING_FACTOR
    alpha = rng * RNG_MULT * np.deg2rad(theta) / RRV_SCALING_FACTOR
    z = 1.0 - (beta**2 / alpha**2)  # beta always <= alpha
    series = hypergeometric_gaussian(-0.3333, 0.5, 2.5, z)
    edr = sw**3 * (1.0/alpha) * (CONSTANT * series)**(-1.5)
    return edr**0.33333


def edr_short_range(sw, rng, theta, gs):
    """
    For gate spacing > range * beamwidth
    sw (spectrum width) in m/s,
    rng (range) in km,
    theta (beamwidth) in deg,
    gs (gate spacing) in km

    """
    beta = gs * RNG_MULT / RRV_SCALING_FACTOR
    alpha = rng * RNG_MULT * np.deg2rad(theta) / RRV_SCALING_FACTOR
    z = 1.0 - (alpha**2 / beta**2)  # alpha always <= beta
    series = hypergeometric_gaussian(-0.3333, 2.0, 2.5, z)
    edr = sw**3 * (1.0/beta) * (CONSTANT * series)**(-1.5)
    return edr**0.33333


def add_turbulence_field(radar, turbulence, turb_name='turbulence'):
    field_dict = {'data': turbulence,
                  'units': 'm^2/3 s^-1',
                  'long_name': 'Cubic Root of Eddy Dissipation Rate',
                  'standard_name': "EDR^1/3"}
    radar.add_field(turb_name, field_dict, replace_existing=True)


def get_sweep_data(radar, field_name, sweep_number):
    # Check if _FillValue exists
    try:
        fill_value = radar.fields[field_name]['_FillValue']
    except KeyError:
        fill_value = BAD_DATA_VAL
    # Check if masked array
    try:
        return radar.fields[field_name]['data'][
            radar.sweep_start_ray_index['data'][sweep_number]:
            radar.sweep_end_ray_index['data'][sweep_number]+1][:].filled(
                fill_value=fill_value)
    except AttributeError:
        return radar.fields[field_name]['data'][
            radar.sweep_start_ray_index['data'][sweep_number]:
            radar.sweep_end_ray_index['data'][sweep_number]+1][:]


def get_sweep_azimuths(radar, sweep_number):
    return radar.azimuth['data'][
        radar.sweep_start_ray_index['data'][sweep_number]:
        radar.sweep_end_ray_index['data'][sweep_number]+1][:]


def get_sweep_elevations(radar, sweep_number):
    return radar.elevation['data'][
        radar.sweep_start_ray_index['data'][sweep_number]:
        radar.sweep_end_ray_index['data'][sweep_number]+1][:]


def calc_cartesian_coords_radians(lon1r, lat1r, lon2r, lat2r):
    """Assumes conversion to radians has already occurred"""
    yy = re * (lat1r - lat2r)
    mlat = (lat1r + lat2r) / 2.0
    xx = re * np.cos(mlat) * (lon1r - lon2r)
    return xx, yy


def get_radar_latlon_plus_radians(radar):
    """Input Py-ART radar object, get lat/lon first in deg then in radians"""
    klat = radar.latitude['data'][0]
    klon = radar.longitude['data'][0]
    klatr = np.deg2rad(klat)
    klonr = np.deg2rad(klon)
    return klat, klon, klatr, klonr


def flatten_and_reduce_data_array(array, condition):
    array = array.ravel()
    return array[condition]

###################################
# INTERNAL FUNCTIONS FOLLOW
###################################


def _calc_turb_rhi_sweep(radar, sweep_number, radius=1.0,
                         verbose=False, name_dz=DEFAULT_DZ,
                         name_sw=DEFAULT_SW, use_ntda=True,
                         beamwidth=DEFAULT_BEAMWIDTH,
                         gate_spacing=DEFAULT_GATE_SPACING):
    """
    radar = Py-ART radar object
    sweep_number = Can be as low as 0, as high as # of sweeps minus 1
    radius = radius of influence (km)
    verbose = Set to True to get more information about calculation progress.
    name_dz = Name of reflectivity field, used by Py-ART to access field
    name_sw = Name of spectrum width field, used by Py-ART to access field
    use_ntda = Flag to use the spatial averaging and weighting employed by NTDA
    beamwidth = Beamwidth of radar in degrees
    gate_spacing = Gate spacing of radar in km
    """
    if verbose:
        overall_time = time.time()
        begin_time = time.time()

    dz_sw = get_sweep_data(radar, name_dz, sweep_number)
    try:
        fill_val_sw = radar.fields[name_sw]['_FillValue']
    except KeyError:
        fill_val_sw = BAD_DATA_VAL
    try:
        fill_val_dz = radar.fields[name_dz]['_FillValue']
    except KeyError:
        fill_val_dz = BAD_DATA_VAL
    sweep_sw = get_sweep_data(radar, name_sw, sweep_number)
    sweep_elev_sw = get_sweep_elevations(radar, sweep_number)

    sw_sr_1d = radar.range['data'][:] / RNG_MULT
    result = np.meshgrid(sw_sr_1d, sweep_elev_sw)
    sw_sr_2d = result[0]
    sw_el = result[1]
    sweep_size = radar.sweep_end_ray_index['data'][sweep_number] + 1 - \
        radar.sweep_start_ray_index['data'][sweep_number]
    xx, yy = rsl_get_groundr_and_h(sw_sr_2d, sw_el)

    if verbose:
        print(time.time() - begin_time,
              'seconds to complete all preliminary processing')
        begin_time = time.time()

    # Determine NTDA interest fields
    csnr_sw = _calc_csnr_for_every_gate(dz_sw, sw_sr_2d)
    crng_sw = _calc_crng_for_every_gate(sw_sr_2d)
    czh_sw = _calc_czh_for_every_gate(dz_sw, yy)
    cpr = 1.0  # Not calculating Cpr, user must remove second trip manually.

    if verbose:
        print(time.time() - begin_time,
              'seconds to precompute Csnr, Crng, Czh')
        begin_time = time.time()

    # Flatten all arrays to avoid performance buzzkill of nested loops.
    # Then reduce the data using masks to make the job manageable.
    sweep_sw = sweep_sw.ravel()
    dz_sw = dz_sw.ravel()
    condition = np.logical_and(dz_sw != fill_val_dz, sweep_sw != fill_val_sw)
    sweep_sw = sweep_sw[condition]
    dz_sw = dz_sw[condition]

    # Brief detour to get eps and check to see if that's all user wants.
    # Placing this code here with reduced sweep_sw improves performance.
    sr_1d = flatten_and_reduce_data_array(sw_sr_2d, condition)
    cond = gate_spacing >= sr_1d * np.deg2rad(beamwidth)
    eps_sw = edr_long_range(sweep_sw, sr_1d, beamwidth, gate_spacing)
    eps_sw[cond] = edr_short_range(sweep_sw[cond], sr_1d[cond],
                                   beamwidth, gate_spacing)
    if not use_ntda:
        turb_radar_f = 1.0 * eps_sw
        turb_radar = sw_sr_2d.flatten() * 0.0 + fill_val_sw
        turb_radar[condition] = turb_radar_f
        eps_sw = np.reshape(turb_radar, (len(sweep_elev_sw), radar.ngates))
        if verbose:
            print(time.time() - overall_time,
                  'seconds to process radar sweep')
        return eps_sw

    # Provided NTDA is wanted, continue flattening/reducing arrays
    xx = flatten_and_reduce_data_array(xx, condition)
    yy = flatten_and_reduce_data_array(yy, condition)
    csnr_sw = flatten_and_reduce_data_array(csnr_sw, condition)
    crng_sw = flatten_and_reduce_data_array(crng_sw, condition)
    czh_sw = flatten_and_reduce_data_array(czh_sw, condition)
    turb_radar_f = 0.0 * sweep_sw + fill_val_sw

    # Find the distance to every other good gate
    ind, ind_sw = _calc_tree(xx, yy, radius)
    cswv_sw = _calc_cswv_for_every_gate(xx, sweep_sw, ind_sw)
    if verbose:
        print(time.time() - begin_time,
              'seconds to get eps, reduce data,',
              'compute BallTree, and get Cswv')
        begin_time = time.time()

    # Loop thru data and do NTDA filtering
    for i in np.arange(len(xx)):
        if verbose:
            if i % 50000 == 0:
                print('i =', i, 'of', len(xx) - 1,
                      time.time() - begin_time, 'seconds elapsed during loop')
        # Broadcating employed to minimize the amount of looping
        eps = eps_sw[ind[i]]**2
        csnr = csnr_sw[ind[i]]**0.6667
        crng = crng_sw[ind[i]]
        czh = czh_sw[ind[i]]
        cswv = cswv_sw[ind[i]]
        # Begin NTDA-specific calculation
        tot = csnr * cpr * cswv * czh * crng
        num = tot * eps
        tot = np.sum(tot)
        num = np.sum(num)
        if tot > 0:
            turb_radar_f[i] = np.sqrt(num/tot)

    # Restore turbulence to a full 2-D sweep array and return along w/ lat/lon.
    turb_radar = sw_sr_2d.flatten() * 0.0 + fill_val_sw
    turb_radar[condition] = turb_radar_f
    turb_radar = np.reshape(turb_radar, (len(sweep_elev_sw), radar.ngates))
    if verbose:
        print(time.time() - overall_time, 'seconds to process radar sweep')
    return turb_radar

###################################


def _initialize_turb_field(radar, name_sw):
    try:
        fill_value = radar.fields[name_sw]['_FillValue']
    except KeyError:
        fill_value = BAD_DATA_VAL
    try:
        turbulence = 0.0 * radar.fields[name_sw]['data'][:].filled(
            fill_value=fill_value) + fill_value
    except AttributeError:
        turbulence = 0.0 * radar.fields[name_sw]['data'][:] + fill_value
    return fill_value, turbulence


def _finalize_turb_field(radar, turbulence, name_dz, name_sw):
    # Combine DZ and SW masks if available
    if hasattr(radar.fields[name_dz]['data'], 'mask'):
        mask1 = radar.fields[name_dz]['data'].mask
    else:
        try:
            fill_val_dz = radar.fields[name_dz]['_FillValue']
        except KeyError:
            fill_val_dz = BAD_DATA_VAL
        mask1 = radar.fields[name_dz]['data'] == fill_val_dz
    if hasattr(radar.fields[name_sw]['data'], 'mask'):
        mask2 = radar.fields[name_sw]['data'].mask
    else:
        mask2 = radar.fields[name_sw]['data'] == fill_value
    combine = np.ma.mask_or(mask1, mask2)
    return np.ma.array(turbulence, mask=combine)


def _retrieve_sweep_fields(radar, name_sw, name_dz, sweep_number,
                           sweep_dz, split_cut):
    if split_cut:
        # Low-level NEXRAD tilts can report DZ and SW from different sweeps
        sweep_sw = get_sweep_data(radar, name_sw, sweep_number+1)
        sweep_az_dz = get_sweep_azimuths(radar, sweep_number)
        sweep_az_sw = get_sweep_azimuths(radar, sweep_number+1)
        sweep_elev_sw = get_sweep_elevations(radar, sweep_number+1)
        dz_sw = 0.0 * sweep_sw
        # Map DZ to SW sweep arrangement
        for inaz1 in np.arange(len(sweep_az_sw)):
            inaz2 = np.argmin(np.abs(sweep_az_sw[inaz1]-sweep_az_dz))
            dz_sw[inaz1][:] = sweep_dz[inaz2][:]
    else:
        sweep_sw = get_sweep_data(radar, name_sw, sweep_number)
        sweep_az_sw = get_sweep_azimuths(radar, sweep_number)
        sweep_elev_sw = get_sweep_elevations(radar, sweep_number)
        dz_sw = 1.0 * sweep_dz
    return sweep_sw, sweep_az_sw, sweep_elev_sw, dz_sw


def _calc_csnr_for_every_gate(dz_sw, sw_sr):
    """TO DO: Turn thresholds into global variables changeable elsewhere"""
    csnr_sw = 0.0 * dz_sw
    snr_sw = dz_sw + 20.0 * np.log10(230.0 / sw_sr)
    condition = np.logical_and(snr_sw >= 10, snr_sw < 20)
    csnr_sw[condition] = 0.1 * (snr_sw[condition] - 10.0)
    condition = np.logical_and(snr_sw >= 20, snr_sw < 70)
    csnr_sw[condition] = 1.0
    condition = np.logical_and(snr_sw >= 70, snr_sw < 80)
    csnr_sw[condition] = 1.0 - 0.1 * (snr_sw[condition] - 70.0)
    return csnr_sw


def _calc_crng_for_every_gate(sw_sr):
    """TO DO: Turn thresholds into global variables changeable elsewhere"""
    crng_sw = 0.0 * sw_sr
    condition = np.logical_and(sw_sr >= 0, sw_sr < 5)
    crng_sw[condition] = 0.2 * (sw_sr[condition] - 5.0)
    condition = np.logical_and(sw_sr >= 5, sw_sr < 140)
    crng_sw[condition] = 1.0
    condition = np.logical_and(sw_sr >= 140, sw_sr < 275)
    crng_sw[condition] = 1.0 - (1.0 / 135.0) * (sw_sr[condition] - 140.0)
    return crng_sw


def _calc_czh_for_every_gate(dz_sw, sw_ht):
    """TO DO: Turn thresholds into global variables changeable elsewhere"""
    czh_sw = 0.0 * dz_sw
    dummy = dz_sw + 3.5 * sw_ht
    condition = np.logical_and(dummy >= 15, dummy < 25)
    czh_sw[condition] = 0.1 * (dummy[condition] - 15.0)
    condition = dummy >= 25
    czh_sw[condition] = 1.0
    return czh_sw


def _calc_cswv_for_every_gate(xx, sweep_sw, ind_sw):
    cswv_sw = 0.0 * xx
    for i in np.arange(len(xx)):
        # Get spectrum width variance via np.dot()
        # Provides ~30% speed improvement over var()
        a = sweep_sw[ind_sw[i]]
        m = a.mean()
        c = a - m
        dummy = np.dot(c, c) / a.size
        cswv_sw[i] = calc_cswv_cython(dummy)
    return cswv_sw


def _calc_tree(xx, yy, radius):
    X = np.zeros((len(xx), 2), dtype='float')
    X[:, 0] = xx[:]
    X[:, 1] = yy[:]
    tree = BallTree(X, metric='euclidean')
    ind = tree.query_radius(X, r=radius)
    ind_sw = tree.query_radius(X, r=VARIANCE_RADIUS_SW)
    return ind, ind_sw
