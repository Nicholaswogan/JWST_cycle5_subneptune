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
            return J_est, est_err
        
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

    Rp1 = unp.nominal_values(Rp)

    # Kempton+2018 Table 1
    scale = np.zeros(len(Rp1))
    scale[np.where(Rp1 < 1.5)] = 0.190
    scale[np.where((Rp1 >= 1.5)  & (Rp1 < 2.75))] = 1.26
    scale[np.where((Rp1 >= 2.75) & (Rp1 < 4.0))] = 1.28
    scale[np.where((Rp1 >= 4.0)  & (Rp1 < 10.0))] = 1.15
    scale[np.where(Rp1 >= 10.0)] = 1.15

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

def  aguichine_Mp_Rp_relation(Mp, x_H2O):
    "Aguichine+2021 relation for steam worlds. Assumes core/(core + mantle) = 30% and T = 400 K"

    _x_H2O = np.array([
        0.10000, 0.20000, 0.30000, 0.40000, 0.50000, 
        0.60000, 0.70000, 0.80000, 0.90000, 1.00000
    ])
    a = np.array([
        0.226906975164, 0.217765024811, 0.212442203827, 0.204976173004, 0.202130265229, 
        0.197084836012, 0.194221621762, 0.189130108899, 0.186645182403, 0.183937966542
    ])
    b = np.array([
        0.094067692688, 0.128018507250, 0.152879500402, 0.178481937058, 0.196448409997, 
        0.216550911456, 0.232352339775, 0.251663613406, 0.266665391910, 0.282211819712
    ])
    c = np.array([
        2.774927261553, 2.783267397986, 2.705782421433, 2.821807208850, 2.736118616554, 
        2.807745472991, 2.727520125379, 2.843993163104, 2.791935041902, 2.797308944390
    ])
    d = np.array([
        1.051566266255, 0.977423150612, 0.944053487148, 0.875932080161, 0.858441722849, 
        0.809922791844, 0.801305433572, 0.747937892614, 0.735664369806, 0.704831238292
    ])
    log10Mp = np.log10(Mp)
    log10Rp = a*log10Mp + np.exp(-d*(log10Mp + c)) + b
    return 10.0**np.interp(x_H2O, _x_H2O, log10Rp)

def determine_skiprows(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        if not line.startswith('#'):
            break
    return i
    
def make_unp_uarray(df, key, inds=None):
    if inds is None:
        inds = np.ones(len(df[key]),bool)
    return unp.uarray(df[key][inds], (df[key+'err1'][inds] - df[key+'err2'][inds])/2)

def set_unp_uarray(df, key, value, inds=None):
    n = len(df[list(df.keys())[0]])
    # Make inds
    if inds is None:
        inds = np.ones(n, bool)
    # Add error keys if needed
    keys = [key, key+'err1', key+'err2']
    for a in keys:
        if a not in df:
            df[a] = np.empty(n)
    df[key][inds] = unp.nominal_values(value[inds])
    df[key+'err1'][inds] = unp.std_devs(value[inds])
    df[key+'err2'][inds] = -unp.std_devs(value[inds])
    return df

def search_nea_csv(filename, filters={}):

    df1 = pd.read_csv(filename, skiprows=determine_skiprows(filename))
    df = {}
    for key in df1:
        df[key] = df1[key].to_numpy()

    # Compute Teq and insolation that is self-consistent
    inds = (
        np.isfinite(df['st_teff']) & 
        np.isfinite(df['st_rad']) & 
        np.isfinite(df['pl_orbsmax'])
    )
    tmp = insolation(make_unp_uarray(df,'st_teff'), make_unp_uarray(df,'st_rad'), make_unp_uarray(df, 'pl_orbsmax')) # W/m^2
    df = set_unp_uarray(df, 'pl_insol', tmp/earth_insolation(), inds) # normalize by Earth
    df = set_unp_uarray(df, 'pl_eqt', equilibrium_temperature(tmp, 0.0), inds)

    # Deal with masses
    # Filter out planets without known radii
    inds = np.isfinite(df['pl_rade']) & ~df['pl_radelim'].astype(bool)
    for key in df:
        df[key] = df[key][inds]

    # Create new mass column
    df['pl_bmasse2'] = np.zeros(len(df['pl_bmasse']))
    df['pl_bmasse2err1'] = np.zeros(len(df['pl_bmasse']))
    df['pl_bmasse2err2'] = np.zeros(len(df['pl_bmasse']))
    df['pl_bmasse2_chen'] = np.zeros(len(df['pl_bmasse']),bool)

    # Where the mass is a limit, we use Kipping mass
    inds = df['pl_bmasselim'].astype(bool)
    df['pl_bmasse2'][inds] = chen_kipping_mass(df['pl_rade'][inds])
    df['pl_bmasse2err1'][inds] = np.zeros(np.sum(inds))*np.nan
    df['pl_bmasse2err2'][inds] = np.zeros(np.sum(inds))*np.nan
    df['pl_bmasse2_chen'][inds] = True
    # Where mass is not a limit, we use the NEA mass
    inds = ~df['pl_bmasselim'].astype(bool)
    df['pl_bmasse2'][inds] = df['pl_bmasse'][inds]
    df['pl_bmasse2err1'][inds] = df['pl_bmasseerr1'][inds]
    df['pl_bmasse2err2'][inds] = df['pl_bmasseerr2'][inds]
    df['pl_bmasse2_chen'][inds] = False

    # Compute density
    pl_rade = make_unp_uarray(df,'pl_rade')
    pl_bmasse2 = make_unp_uarray(df,'pl_bmasse2')
    dene = (constants.M_earth.value)/((4/3)*np.pi*(constants.R_earth.value)**3)
    pl_dene = (pl_bmasse2*constants.M_earth.value)/((4/3)*np.pi*(pl_rade*constants.R_earth.value)**3)
    df = set_unp_uarray(df, 'pl_dene', pl_dene/dene)

    # Filter out stars with unknown magnitudes, Teq
    inds = np.isfinite(df['sy_vmag']) & np.isfinite(df['sy_kmag']) & np.isfinite(df['pl_eqt']) & np.isfinite(df['st_rad'])
    for key in df:
        df[key] = df[key][inds]

    # Estimate the J mag from V and Ks
    mamajek = MamajekColorTable()
    tmp = np.array([mamajek.estimate_J_from_V_Ks(df['sy_vmag'][i], df['sy_kmag'][i]) for i in range(len(df['sy_kmag']))])
    df['sy_jmag'] = tmp[:,0]
    df['sy_jmagerr1'] = tmp[:,1]
    df['sy_jmagerr2'] = -tmp[:,1]

    # TSM
    pl_tsm = compute_TSM(
        make_unp_uarray(df,'pl_rade'), 
        make_unp_uarray(df,'pl_eqt'), 
        make_unp_uarray(df,'pl_bmasse2'), 
        make_unp_uarray(df,'st_rad'), 
        make_unp_uarray(df,'sy_jmag')
    )
    df = set_unp_uarray(df, 'pl_tsm', pl_tsm)

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
            if pl_name not in list(df['pl_name']):
                ind = list(df_save['pl_name']).index(pl_name)
                for key in df:
                    df[key] = np.append(df[key],df_save[key][ind])

    return df