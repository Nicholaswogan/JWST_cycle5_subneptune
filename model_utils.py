import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import interpolate
from picaso import justdoit as jdi
from astropy import constants
from scipy import constants as const
from photochem.utils import stars
from photochem.extensions import gasgiants
from photochem._clima import rebin, rebin_with_errors
from photochem.equilibrate import ChemEquiAnalysis
import astropy.units as u
import pickle
import os
import re
from copy import deepcopy

def make_outfile_name(name, mh, CtoO, tint):
    outfile = name+'_MH=%.3f_CO=%.3f_Tint=%.1f.pkl'%(mh, CtoO, tint)
    return outfile

class ClimateHJ():

    def __init__(self, planet_mass, planet_radius, P_ref, Teq, 
                 T_star, logg_star, metal_star, r_star, database_dir):
        """Initialized the climate model.

        Parameters
        ----------
        planet_mass : float
            Planet mass in Earth masses
        planet_radius : float
            Plane radius in Earth radii
        P_ref : float
            Reference pressure in dynes/cm^2
        semi_major : float
            Semi-major axis in AU
        Teq : float
            Equilibrium temperature in K
        T_star : float
            Stellar effective temperature in K
        logg_star : float
            Stellar gravity in logg
        metal_star : float
            Stellar metallicity in log10 units
        r_star : float
            Stellar radius in solar radii
        database_dir : str
            Path to where climate opacities are stored.
        """        
        self.nlevel = 91
        self.nofczns = 1
        self.nstr_upper = 85
        self.rfacv = 0.5 
        self.p_bottom = 3 # log10(bars)
        self.planet_mass = planet_mass # Earth masses
        self.planet_radius = planet_radius # Earth radii
        self.P_ref = P_ref # dynes/cm^2
        self.Teq = Teq # K
        self.T_star = T_star # K

        Fp = stars.equilibrium_temperature_inverse(Teq,0)
        Fs = stars.stefan_boltzmann(T_star)
        Rs = r_star*constants.R_sun.value

        # Compute based on Teq and stellar properties
        self.semi_major = np.sqrt(Fs*np.pi*Rs**2/(np.pi*Fp))/constants.au.value # AU
        self.logg_star = logg_star
        self.metal_star = metal_star # log10 metallicity
        self.r_star = r_star # solar radii
        self.database_dir = database_dir
        self.outfolder = './'

        # 
        self.opacity_ck = None

    def run_climate_model(self, metallicity, CtoO, tint, save_output=False, case_name=None):
        """Runs the climate model.

        Parameters
        ----------
        metallicity : float
            Metallicity relative to solar
        CtoO : float
            The C/O ratio relative to solar
        tint : float
            Intrinsic temperature
        save_output : bool, optional
            If True, the the output is saved to a pickle file, by default False

        Returns
        -------
        dict
            Dictionary containing the P-T profile
        """

        if save_output:
            if case_name is None:
                raise Exception('case_name must be specified')

        # Get the opacity database
        mh = np.log10(metallicity)
        if mh >= 0:   
            mh_str = ('+%.2f'%mh).replace('.','')
        else:
            mh_str = ('-%.2f'%mh).replace('.','')
        CtoO_str = ('%.2f'%CtoO).replace('.','')

        ck_db = self.database_dir+f'sonora_2020_feh{mh_str}_co_{CtoO_str}.data.196'
        self.opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted')
        
        # Initialize climate run
        cl_run = jdi.inputs(calculation="planet", climate = True)

        cl_run.inputs['approx']['p_reference'] = self.P_ref/1e6

        # set gravity
        grav = gasgiants.gravity(self.planet_radius*constants.R_earth.value*1e2, self.planet_mass*constants.M_earth.value*1e3, 0.0)/1e2
        cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
        
        # Set tint
        cl_run.effective_temp(tint) 

        # Set stellar properties
        T_star = self.T_star
        logg = self.logg_star #logg, cgs
        metal = self.metal_star # metallicity of star
        r_star = self.r_star # solar radius
        semi_major = self.semi_major # star planet distance, AU
        cl_run.star(self.opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star,
                    radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU, database='phoenix')

        # Initial temperature guess
        nlevel = self.nlevel # number of plane-parallel levels in your code
        Teq = self.Teq # planet equilibrium temperature
        pt = cl_run.guillot_pt(Teq, nlevel=nlevel, T_int=tint, p_bottom=self.p_bottom, p_top=-6)
        temp_guess = pt['temperature'].values
        pressure = pt['pressure'].values

        nofczns = self.nofczns # number of convective zones initially. Let's not play with this for now.
        nstr_upper = self.nstr_upper # top most level of guessed convective zone
        nstr_deep = nlevel - 2 # this is always the case. Dont change this
        nstr = np.array([0,nstr_upper,nstr_deep,0,0,0]) # initial guess of convective zones
        rfacv = self.rfacv

        # Set inputs
        cl_run.inputs_climate(
            temp_guess=temp_guess, 
            pressure=pressure,
            nstr=nstr, 
            nofczns=nofczns , 
            rfacv=rfacv
        )

        # Run model
        out = cl_run.climate(self.opacity_ck)

        if save_output:
            outfile = os.path.join(self.outfolder,make_outfile_name(case_name, mh, CtoO, tint))
            with open(outfile,'wb') as f:
                pickle.dump(out,f)

        return out
    

def get_comp_yang(metallicity, x_acc=None, CtoO=None, N_depletion=None):
    """Yang and Hu (2024) method for computing atomic composition.
    x_acc is [H2]/([H2] + [H2O]). So 
    """

    if N_depletion is not None and x_acc is not None:
        raise Exception('')
    if CtoO is not None and x_acc is not None:
        raise Exception('')
    
    comp = {
        'O': 0.000495,
        'C': 0.000272,
        'N': 0.000065,
        'S': 0.000013,
        'He': 0.077379,
        'H': 0.921775
    }
    tot = sum(comp.values())
    for key in comp:
        comp[key] /= tot
    for key in comp:
        if key not in ['H','He']:
            comp[key] *= metallicity
    tot = sum(comp.values())
    for key in comp:
        comp[key] /= tot

    # Acreete H2O
    if x_acc is not None:
        a = comp['H'] + comp['He'] + comp['O']
        comp['H'] = (2*a)/(2 + (1 - x_acc) + 2/11.91)
        comp['He'] = (2*a)/((2 + (1 - x_acc) + 2/11.91)*11.91)
        comp['O'] = a*(1 - x_acc)/(2 + (1 - x_acc) + 2/11.91)

    # Apply C/O ratio
    if CtoO is not None:
        x = CtoO*(comp['C']/comp['O'])
        a = (x*comp['O'] - comp['C'])/(1 + x)
        comp['C'] = comp['C'] + a
        comp['O'] = comp['O'] - a

    # Deplete N
    if N_depletion is not None:
        comp['N'] /= N_depletion
        
    tot = sum(comp.values())
    for key in comp:
        comp[key] /= tot

    return comp

def get_comp(metallicity, CtoO=None, N_depletion=None, CtoO_conserve_metals=True):

    # From equilibrate
    comp = {
        'H': 0.9208497219145748,
        'N': 6.225707897826845e-05,
        'O': 0.00045101266495106745,
        'C': 0.0002478498773072209,
        'S': 1.2139161871095582e-05,
        'He': 0.07837701930231748
    }

    # Apply metallicity
    for key in comp:
        if key not in ['H','He']:
            comp[key] *= metallicity
    tot = sum(comp.values())
    for key in comp:
        comp[key] /= tot

    # Apply C/O ratio
    if CtoO is not None:
        if CtoO_conserve_metals:
            x = CtoO*(comp['C']/comp['O'])
            a = (x*comp['O'] - comp['C'])/(1 + x)
            comp['C'] = comp['C'] + a
            comp['O'] = comp['O'] - a
        else:
            comp['O'] = comp['C']/(CtoO*(comp['C']/comp['O']))
        # Normalize
        tot = sum(comp.values())
        for key in comp:
            comp[key] /= tot

    # N depletion
    if N_depletion is not None:
        comp['N'] /= N_depletion
        tot = sum(comp.values())
        for key in comp:
            comp[key] /= tot

    return comp
    
def initialize_photochem(spectrum, planet_mass, planet_radius, metallicity, x_acc, CtoO, N_depletion, climate_filename, Kzz):

    pc = gasgiants.EvoAtmosphereGasGiant(
        'inputs/photochem_rxns.yaml',
        spectrum,
        planet_mass*constants.M_earth.value*1e3, # grams
        planet_radius*constants.R_earth.value*1e2, # cm
        solar_zenith_angle=60,
        thermo_file='inputs/photochem_thermo.yaml'
    )
    pc.gdat.verbose = True
    pc.var.verbose = 1
    pc.gdat.TOA_pressure_avg = 0.01

    # Set particle radius
    particle_radius = pc.var.particle_radius
    particle_radius[:,:] = 1e-4
    pc.var.particle_radius = particle_radius
    pc.update_vertical_grid(TOA_alt=pc.var.top_atmos)

    # Set composition
    comp = get_comp_yang(metallicity, x_acc=x_acc, CtoO=CtoO, N_depletion=N_depletion)
    molfracs_atoms = np.empty(len(pc.gdat.gas.atoms_names))
    for i,atom in enumerate(pc.gdat.gas.atoms_names):
        molfracs_atoms[i] = comp[atom]
    pc.gdat.gas.molfracs_atoms_sun = molfracs_atoms

    with open(climate_filename,'rb') as f:
        out = pickle.load(f)
    P = out['pressure'][::-1].copy()*1e6
    T = out['temperature'][::-1].copy()
    if np.max(P) <= 1e10:
        P_append = np.arange(np.log10(np.max(P)), 10.01,.1)[::-1][:-1]
        T1 = interpolate.interp1d(np.log10(P[::-1]), T[::-1], fill_value='extrapolate')(P_append)
        P = np.append(10.0**P_append,P)
        T = np.append(T1,T)
    Kzz1 = np.ones(P.shape[0])*Kzz

    pc.initialize_to_climate_equilibrium_PT(P, T, Kzz1, 1, 1)

    return pc

def return_atmosphere_picaso(pc):
    sol = pc.return_atmosphere()
    # P and T
    out = {
        'pressure': sol['pressure']/1e6, # to bars
        'temperature': sol['temperature']
    }
    # Mixing ratios
    species_names = pc.dat.species_names[pc.dat.np:(-2-pc.dat.nsl)]
    for key in species_names:
        if key not in ['pressure','temperature','Kzz']:
            out[key] = sol[key]
    # Flip
    for key in out:
        out[key] = out[key][::-1].copy()
    return out

def run_photochem(outdir, case_name, spectrum, planet_mass, planet_radius, metallicity, x_acc, CtoO, N_depletion, climate_filename, Kzz):

    # Initialize
    pc = initialize_photochem(spectrum, planet_mass, planet_radius, metallicity, x_acc, CtoO, N_depletion, climate_filename, Kzz)

    # Find steady state
    assert pc.find_steady_state()

    # Save model state so we can reopen
    state = pc.model_state_to_dict()
    filename = os.path.join(outdir, case_name+'_state.pkl')
    with open(filename,'wb') as f:
        pickle.dump(state,f)

    # Save in PICASO format
    sol = return_atmosphere_picaso(pc)
    filename = os.path.join(outdir, case_name+'_picaso.pkl')
    with open(filename,'wb') as f:
        pickle.dump(sol,f)

class Picaso():

    def __init__(self, filename_db, M_planet, R_planet, R_star, opannection_kwargs={}, star_kwargs={}):

        self.opa = jdi.opannection(filename_db=filename_db, **opannection_kwargs)
        self.case = jdi.inputs()
        self.case.phase_angle(0)
        self.case.gravity(mass=M_planet, mass_unit=jdi.u.Unit('M_earth'),
                     radius=R_planet, radius_unit=jdi.u.Unit('R_earth'))
        self.case.star(self.opa, radius=R_star, radius_unit=jdi.u.Unit('R_sun'), **star_kwargs)

    def clouds_reset(self):
        self.case.inputs['clouds'] = {
            'profile': None,
            'wavenumber': None,
            'scattering': {'g0': None, 'w0': None, 'opd': None}
        }

    def _spectrum(self, atm, calculation='thermal', atmosphere_kwargs={}, log10Pcloudbottom=None, dlog10Pcloud=None, **kwargs):
        self.case.atmosphere(df=atm, verbose=False, **atmosphere_kwargs)
        if log10Pcloudbottom is not None or log10Pcloudbottom is not None:
            self.case.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[log10Pcloudbottom], dp=[dlog10Pcloud])
        df = self.case.spectrum(self.opa, calculation=calculation, **kwargs)
        return df

    def rprs2(self, atm, R=100, wavl=None, atmosphere_kwargs={},log10Pcloudbottom=None, dlog10Pcloud=None, **kwargs):

        df = self._spectrum(
            atm, 'transmission', atmosphere_kwargs=atmosphere_kwargs, 
            log10Pcloudbottom=log10Pcloudbottom, dlog10Pcloud=dlog10Pcloud, 
            **kwargs
        )

        wv_h = 1e4/df['wavenumber'][::-1].copy()
        wavl_h = stars.make_bins(wv_h)
        rprs2_h = df['transit_depth'][::-1].copy()

        if wavl is None:
            wavl = stars.grid_at_resolution(np.min(wavl_h), np.max(wavl_h), R)

        rprs2 = rebin(wavl_h.copy(), rprs2_h.copy(), wavl.copy())

        return wavl, rprs2, df
    
def create_exo_dict(R_planet, R_star, total_observing_time, eclipse_duration, kmag, starpath):
    from pandexo.engine import justdoit as pandexo_jdi

    exo_dict = pandexo_jdi.load_exo_dict()

    exo_dict['observation']['sat_level'] = 80
    exo_dict['observation']['sat_unit'] = '%'
    exo_dict['observation']['noccultations'] = 1
    exo_dict['observation']['R'] = None
    exo_dict['observation']['baseline_unit'] = 'total'
    exo_dict['observation']['baseline'] = total_observing_time
    exo_dict['observation']['noise_floor'] = 0

    exo_dict['star']['type'] = 'user'
    exo_dict['star']['mag'] = kmag
    exo_dict['star']['ref_wave'] = 2.22
    exo_dict['star']['starpath'] = starpath
    exo_dict['star']['w_unit'] = 'um'
    exo_dict['star']['f_unit'] = 'FLAM'
    exo_dict['star']['radius'] = R_star
    exo_dict['star']['r_unit'] = 'R_sun'

    exo_dict['planet']['type'] = 'constant'
    exo_dict['planet']['transit_duration'] = eclipse_duration
    exo_dict['planet']['td_unit'] = 's'
    exo_dict['planet']['radius'] = R_planet
    exo_dict['planet']['r_unit'] = 'R_earth'
    exo_dict['planet']['f_unit'] = 'rp^2/r*^2'

    return exo_dict

def _run_pandexo(R_planet, R_star, total_observing_time, eclipse_duration, kmag, inst, starpath, verbose=False, **kwargs):
    from pandexo.engine import justdoit as pandexo_jdi

    exo_dict = create_exo_dict(R_planet, R_star, total_observing_time, eclipse_duration, kmag, starpath)

    # Run pandexo
    result = pandexo_jdi.run_pandexo(exo_dict, inst, verbose=verbose, **kwargs)

    return result

def run_pandexo(R_planet, R_star, total_observing_time, eclipse_duration, kmag, inst, starpath, R=None, dwv=None, ntrans=1, verbose=False, **kwargs):

    # inst is just a string
    if isinstance(inst, str):
        inst = [inst]
    elif isinstance(inst, dict):
        pass
    else:
        raise Exception()
    
    result = _run_pandexo(R_planet, R_star, total_observing_time, eclipse_duration, kmag, inst, starpath, verbose, **kwargs)

    spec = result['FinalSpectrum']
    wavl = stars.make_bins(spec['wave'])
    F = spec['spectrum']
    err = spec['error_w_floor']
    err = err/np.sqrt(ntrans)

    if R is not None and dwv is not None:
        raise Exception()

    if R is not None:
        wavl_n = stars.grid_at_resolution(np.min(wavl), np.max(wavl), R)
        F_n, err_n = rebin_with_errors(wavl.copy(), F.copy(), err.copy(), wavl_n.copy())
        wavl = wavl_n
        F = F_n
        err = err_n

    if dwv is not None:
        wavl_n = np.arange(np.min(wavl), np.max(wavl), dwv)
        F_n, err_n = rebin_with_errors(wavl.copy(), F.copy(), err.copy(), wavl_n.copy())
        wavl = wavl_n
        F = F_n
        err = err_n

    return wavl, F, err, result

def wv_bins_from_wavl(wavl):
    wv_bins = np.empty((2,len(wavl)-1))
    for i in range(len(wavl)-1):
        wv_bins[0,i] = wavl[i]
        wv_bins[1,i] = wavl[i+1]
    return wv_bins

def interp_spectrum_to_data(wv_bins, wavl, rprs2):
    res = np.empty(wv_bins.shape[1])
    for i in range(wv_bins.shape[1]):
        res[i] = rebin(wavl.copy(), rprs2.copy(), wv_bins[:,i].copy())[0]
    return res

def species_to_latex(sp):
    sp1 = re.sub(r'([0-9]+)', r"_\1", sp)
    sp1 = r'$\mathrm{'+sp1+'}$'
    return sp1

def synthetic_data_pandexo(pl, st, starpath, insts):
    
    pandexo_res = []
    for j, inst in enumerate(insts):
        if isinstance(inst, tuple):
            inst_name, inst = inst
        else:
            inst_name = inst
        wavl_d, rprs_d, err_d, result_d = run_pandexo(
            pl.radius,
            st.radius,
            pl.transit_duration*2,
            pl.transit_duration,
            st.kmag,
            inst=inst,
            starpath=starpath,
            ntrans=1,
            R=None,
            dwv=None
        )
        pandexo_res.append((wavl_d, rprs_d, err_d, result_d))

    return pandexo_res

def synthetic_data(pandexo_res, insts, R, dwv, ntrans, inflation, wavl_true, rprs_true, random):
    
    data_dicts = {}
    for j, inst in enumerate(insts):
        if isinstance(inst, tuple):
            inst_name, inst = inst
        else:
            inst_name = inst
        wavl_d, _, err_d, result_d = deepcopy(pandexo_res[j])

        # Number of transits
        err_d = err_d/np.sqrt(ntrans[j])

        # Inflate errors
        err_d *= inflation

        # Make the fake data
        rprs2_d = interp_spectrum_to_data(wv_bins_from_wavl(wavl_d), wavl_true, rprs_true)

        # Add randomness at native resolution
        if random:
            rprs2_d += err_d*(np.random.randn(len(err_d)))

        # Now we downbin to desired resolution
        if R[j] is not None and dwv[j] is not None:
            raise Exception()
        if R[j] is not None:
            wavl_n = stars.grid_at_resolution(np.min(wavl_d), np.max(wavl_d), R[j])
        if dwv[j] is not None:
            wavl_n = np.arange(np.min(wavl_d), np.max(wavl_d), dwv[j])
        if R[j] is not None or dwv[j] is not None:
            rprs2_n, err_n = rebin_with_errors(wavl_d.copy(), rprs2_d.copy(), err_d.copy(), wavl_n.copy())
            wavl_d = wavl_n
            rprs2_d = rprs2_n
            err_d = err_n

        # Derived quantities
        wv_bins_d = wv_bins_from_wavl(wavl_d)
        wv = np.mean(wv_bins_d,axis=0)
        wv_err = (wv_bins_d[1,:] - wv_bins_d[0,:])/2

        if inst_name == 'NIRSpec G395H':
            inds = np.where(err_d > 0.01)[0]
            end = inds[0] - 1
            start = inds[-1] + 1

            data_dicts['NRS1'] = {
                'wv_bins': wv_bins_d[:,:end],
                'wv': wv[:end],
                'wv_err': wv_err[:end],
                'rprs2': rprs2_d[:end],
                'err': err_d[:end],
                'result': result_d
            }
            data_dicts['NRS2'] = {
                'wv_bins': wv_bins_d[:,start:],
                'wv': wv[start:],
                'wv_err': wv_err[start:],
                'rprs2': rprs2_d[start:],
                'err': err_d[start:],
                'result': result_d
            }
        else:
            data_dicts[inst_name] = {
                'wv_bins': wv_bins_d,
                'wv': wv,
                'wv_err': wv_err,
                'rprs2': rprs2_d,
                'err': err_d,
                'result': result_d
            }
    return data_dicts

def H2_scale_height_conversion(pl, st, mubar=2.3):
    Rp = pl.radius*constants.R_earth.to('cm').value
    Rs = st.radius*constants.R_sun.to('cm').value
    Mp = pl.mass*constants.M_earth.to('g').value
    T = pl.Teq
    g = gasgiants.gravity(Rp, Mp, 0.0)
    H = (const.N_A*const.k*1e7*T)/(mubar*g)
    A = 2*Rp*H/Rs**2
    return A

