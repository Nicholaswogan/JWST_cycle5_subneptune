import numpy as np
from uncertainties import unumpy as unp
from matplotlib import pyplot as plt
import urllib.request
import os

from scipy import constants as const
from astropy import constants

from picaso import fluxes
import pandas as pd
from copy import deepcopy

class MamajekColorTable:
    """
    Class to manage Mamajek dwarf color table and estimate J magnitudes.
    """
    DEFAULT_URL = "https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
    
    def __init__(self, cache_file="mamajek_colors.txt", url=None):
        self.cache_file = cache_file
        self.url = url or self.DEFAULT_URL

        if not os.path.exists(self.cache_file):
            print(f"Downloading Mamajek color table from {self.url} ...")
            urllib.request.urlretrieve(self.url, self.cache_file)
            print(f"Saved to {self.cache_file}")

        self._VminusKs, self._JminusKs = self._load_table()
    
    def _load_table(self):
        """
        Parse Mamajek table to extract V-Ks and J-Ks relations.
        """
        with open(self.cache_file, "r") as f:
            lines = f.readlines()

        header_idx = None
        for i, line in enumerate(lines[:40]):
            if 'V-Ks' in line and 'J-H' in line and 'H-Ks' in line:
                header_idx = i
                break
        if header_idx is None:
            raise RuntimeError("Could not locate header line in Mamajek table.")

        cols = lines[header_idx].split()
        i_VmKs = cols.index('V-Ks')
        i_JmH  = cols.index('J-H')
        i_HmKs = cols.index('H-Ks')

        data = []
        for line in lines[header_idx+1:]:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) <= max(i_VmKs, i_JmH, i_HmKs):
                continue
            try:
                vkm = float(parts[i_VmKs])
                jh  = float(parts[i_JmH])
                hkm = float(parts[i_HmKs])
                data.append((vkm, jh, hkm))
            except ValueError:
                continue

        data = np.array(data)
        VminusKs = data[:,0]
        JminusKs = data[:,1] + data[:,2]  # J-Ks = (J-H) + (H-Ks)
        order = np.argsort(VminusKs)
        return VminusKs[order], JminusKs[order]

    def estimate_J_from_V_Ks(self, V_mag, Ks_mag, return_components=False):
        """
        Estimate J magnitude from V and Ks (assumes dwarf star, low extinction).
        """
        VminusKs = np.atleast_1d(V_mag - Ks_mag).astype(float)
        JminusKs_interp = np.interp(VminusKs, self._VminusKs, self._JminusKs,
                                    left=self._JminusKs[0], right=self._JminusKs[-1])
        J_est = np.atleast_1d(Ks_mag) + JminusKs_interp

        # crude error estimate
        est_err = np.full_like(J_est, 0.05, dtype=float)

        if J_est.size == 1:
            J_est = J_est.item()
            JminusKs_interp = JminusKs_interp.item()
            est_err = est_err.item()

        if return_components:
            return J_est, {"J-Ks": JminusKs_interp, "est_err_mag": est_err}
        else:
            return J_est
        
def earth_insolation():
    # Get W/m^2 at Earth
    F_s_sun = constants.L_sun.value/(4*np.pi*constants.R_sun.value**2)
    F_p_earth = F_s_sun*((constants.R_sun.value)/(constants.au.value))**2
    return F_p_earth

def insolation(Teff, Rs, a):
    # Compute insolation at planet in W/m^2
    F_s = const.sigma*Teff**4
    F_p = F_s*((Rs*constants.R_sun.value)/(a*constants.au.value))**2
    return F_p

def equilibrium_temperature(stellar_radiation, bond_albedo):
    T_eq = ((stellar_radiation*(1.0 - bond_albedo))/(4.0*const.sigma))**(0.25)
    return T_eq 

def escape_velocity(mass, radius):
    Mp = mass*constants.M_earth.value
    Rp = radius*constants.R_earth.value
    return np.sqrt(2*const.G*Mp/Rp)

def chen_kipping_mass(Rp):
    """
    Chen & Kipping (2017) empirical piecewise mass-radius used in Kempton+2018 (as implemented by Louie+2018).
    Input:
      Rp : planet radius in Earth radii
    Returns:
      Mp : planet mass in Earth masses
    """
    Mp = np.zeros(len(Rp))
    inds = np.where(Rp < 1.23)
    Mp[inds] = 0.9718 * Rp[inds]**3.58
    inds = np.where(Rp >= 1.23)
    Mp[inds] = 1.436 * Rp[inds]**1.70
    return Mp
    
def compute_TSM(Rp, Teq, Mp, Rs, m_j):
    "Compute the Transmission Spectroscopy Metric (TSM) from Kempton et al. (2018)"

    # Kempton+2018 Table 1
    scale = np.zeros(len(Rp))
    scale[np.where(Rp < 1.5)] = 0.190
    scale[np.where((Rp >= 1.5)  & (Rp < 2.75))] = 1.26
    scale[np.where((Rp >= 2.75) & (Rp < 4.0))] = 1.28
    scale[np.where((Rp >= 4.0)  & (Rp < 10.0))] = 1.15
    scale[np.where(Rp >= 10.0)] = 1.15

    factor_mag = 10.0**(-m_j/5)
    TSM = scale * (Rp**3.0*Teq)/(Mp*Rs**2.0) * factor_mag

    return TSM

def print_planet_properties(df, name):
    ind = list(df['pl_name']).index(name)
    for key in df:
        val = df[key][ind]
        print('{:15}'.format(key)+'{:15}'.format(val))

def zeng_Mp_Rp_relation(Mp, CMF):
    "Zeng 2016 relation between Mp and Rp"
    Rp = (1.07 - 0.21*CMF)*(Mp)**(1/3.7)
    return Rp

def determine_skiprows(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        if not line.startswith('#'):
            break
    return i
    
def search_nea_csv(filename, filters={}):

    df1 = pd.read_csv(filename, skiprows=determine_skiprows(filename))
    df = {}
    for key in df1:
        df[key] = df1[key].to_numpy()

    # If pl_eqt is not specified, then compute it
    inds = (
        np.isnan(df['pl_eqt']) & 
        np.isfinite(df['st_teff']) & 
        np.isfinite(df['st_rad']) & 
        np.isfinite(df['pl_orbsmax'])
    )
    tmp = insolation(df['st_teff'][inds], df['st_rad'][inds], df['pl_orbsmax'][inds]) # W/m^2
    df['pl_insol'][inds] = tmp/earth_insolation() # normalize by Earth
    df['pl_eqt'][inds] = equilibrium_temperature(tmp, 0.0)

    # Same exercise for pl_insol
    inds = (
        np.isnan(df['pl_insol']) & 
        np.isfinite(df['st_teff']) & 
        np.isfinite(df['st_rad']) & 
        np.isfinite(df['pl_orbsmax'])
    )
    tmp = insolation(df['st_teff'][inds], df['st_rad'][inds], df['pl_orbsmax'][inds]) # W/m^2
    df['pl_insol'][inds] = tmp/earth_insolation() # normalize by Earth
    df['pl_eqt'][inds] = equilibrium_temperature(tmp, 0.0)

    # Deal with masses
    # Filter out planets without known radii
    inds = np.isfinite(df['pl_rade']) & ~df['pl_radelim'].astype(bool)
    for key in df:
        df[key] = df[key][inds]

    # Create new mass column
    df['pl_bmasse2'] = np.zeros(len(df['pl_bmasse']))
    df['pl_bmasse2_chen'] = np.zeros(len(df['pl_bmasse']),bool)

    # Where the mass is a limit, we use Kipping mass
    inds = df['pl_bmasselim'].astype(bool)
    df['pl_bmasse2'][inds] = chen_kipping_mass(df['pl_rade'][inds])
    df['pl_bmasse2_chen'][inds] = True
    # Where mass is not a limit, we use the NEA mass
    inds = ~df['pl_bmasselim'].astype(bool)
    df['pl_bmasse2'][inds] = df['pl_bmasse'][inds]
    df['pl_bmasse2_chen'][inds] = False

    # Filter out stars with unknown magnitudes, Teq
    inds = np.isfinite(df['sy_vmag']) & np.isfinite(df['sy_kmag']) & np.isfinite(df['pl_eqt']) & np.isfinite(df['st_rad'])
    for key in df:
        df[key] = df[key][inds]

    # Estimate the J mag from V and Ks
    mamajek = MamajekColorTable()
    df['sy_jmag'] = np.array([mamajek.estimate_J_from_V_Ks(df['sy_vmag'][i], df['sy_kmag'][i]) for i in range(len(df['sy_kmag']))])

    # TSM
    df['pl_tsm'] = compute_TSM(df['pl_rade'], df['pl_eqt'], df['pl_bmasse2'], df['st_rad'], df['sy_jmag'])
    inds = np.isfinite(df['pl_tsm'])
    for key in df:
        df[key] = df[key][inds]

    df = apply_filters_to_nea(df, filters)

    return df

def jwst_observations(filename, nea_filename, filters={}, uniques=True):

    df1 = pd.read_csv(filename)
    df = {}
    for key in df1:
        df[key] = df1[key].to_numpy()

    # Must have NEA name
    inds = df['NEA name'] == df['NEA name']
    for key in df:
        df[key] = df[key][inds]

    # Get pl_name
    planet_names = []
    for i in range(len(df['NEA name'])):
        planet_names.append(str(df['NEA name'][i]) +' '+ df['Planet letter'][i])
    df['pl_name'] = np.array(planet_names)

    # Trim the keys
    keys = ['pl_name','Program','PI name','Event','Mode','urls']
    df1 = {}
    for key in keys:
        df1[key] = df[key]
    df = df1

    # Add NEA information to each observation
    df_nea = search_nea_csv(nea_filename)
    for key in df_nea:
        if key != 'pl_name':
            df[key] = []
    for i in range(len(df['pl_name'])):
        ind = list(df_nea['pl_name']).index(df['pl_name'][i])
        for key in df_nea:
            if key != 'pl_name':
                df[key].append(df_nea[key][ind])
    for key in df_nea:
        if key != 'pl_name':
            df[key] = np.array(df[key])

    if uniques:
        # Get Uniques
        _, inds = np.unique(df['pl_name'],return_index=True)
        for key in df:
            df[key] = df[key][inds]

    df = apply_filters_to_nea(df, filters)

    return df

def apply_filters_to_nea(df, filters):

    df_save = deepcopy(df)

    # Apply filters
    for f in filters:
        # Special filters are skipped
        if f in ['pl_bmasseerr_tol','pl_radeerr_tol','pl_name_keep']:
            continue

        # Must be specified
        try:
            inds = np.where(np.isfinite(df[f]))
            for key in df:
                df[key] = df[key][inds]
        except TypeError:
            pass 
        if isinstance(filters[f],list):
            # Range
            inds = (df[f] >= filters[f][0]) & (df[f] <= filters[f][1])
        else:
            # Must be equal
            inds = df[f] == filters[f]
        for key in df:
            df[key] = df[key][inds]

    if 'pl_bmasseerr_tol' in filters:
        mean_error = (df['pl_bmasseerr1'] - df['pl_bmasseerr2'])/2
        frac_error = mean_error/df['pl_bmasse']
        inds = frac_error < filters['pl_bmasseerr_tol']
        for key in df:
            df[key] = df[key][inds]

    if 'pl_radeerr_tol' in filters:
        mean_error = (df['pl_radeerr1'] - df['pl_radeerr2'])/2
        frac_error = mean_error/df['pl_rade']
        inds = frac_error < filters['pl_radeerr_tol']
        for key in df:
            df[key] = df[key][inds]

    # Keepers
    if 'pl_name_keep' in filters:
        for pl_name in filters['pl_name_keep']:
            ind = list(df_save['pl_name']).index(pl_name)
            for key in df:
                df[key] = np.append(df[key],df_save[key][ind])

    return df