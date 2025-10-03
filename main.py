import warnings
warnings.filterwarnings("ignore")
import planets
import model_utils
from copy import deepcopy
import numpy as np
import pickle
import pandas as pd
import os

def climate():

    st = planets.TOI1266
    names = ['TOI1266c']
    pls = [planets.TOI1266c]
    for i,pl in enumerate(pls):
        
        c = model_utils.ClimateHJ(
            pl.mass,
            pl.radius,
            1e6,
            pl.Teq,
            st.Teff,
            st.logg,
            st.metal,
            st.radius,
            database_dir='/Users/nicholas/Applications/picaso_data/climate/'
        )
        c.outfolder = './results/'

        c.run_climate_model(
            metallicity=100, 
            CtoO=1, 
            tint=25,
            save_output=True,
            case_name=names[i]
        )

def photochemistry():

    pl = planets.TOI1266c
    inputs_nominal = {
        'outdir': 'results/',
        'case_name': None,
        'spectrum': 'inputs/TOI1266c_spectrum.txt',
        'planet_mass': pl.mass,
        'planet_radius': pl.radius,
        'metallicity': None,
        'x_acc': None,
        'CtoO': None,
        'N_depletion': None,
        'climate_filename': 'results/TOI1266c_MH=2.000_CO=1.000_Tint=45.0.pkl',
        'Kzz': 1e7
    }
    
    # Escape increases planet b metallicity
    inputs = deepcopy(inputs_nominal)
    inputs['case_name'] = 'TOI1266c_Model1'
    inputs['metallicity'] = 50
    model_utils.run_photochem(**inputs)

    # Magma ocean interaction decreases planet c C/O ratio, and depletes NH3
    inputs = deepcopy(inputs_nominal)
    inputs['case_name'] = 'TOI1266c_Model2'
    inputs['metallicity'] = 50
    inputs['N_depletion'] = 1000
    inputs['CtoO'] = 0.1 # based on Werlen
    model_utils.run_photochem(**inputs)

    # Ice accretion makes both planets H2O-rich like GJ 9827 d
    inputs = deepcopy(inputs_nominal)
    inputs['case_name'] = 'TOI1266c_Model3'
    inputs['metallicity'] = 10
    inputs['x_acc'] = 0.5
    model_utils.run_photochem(**inputs)

    # TOI1266b
    pl = planets.TOI1266b
    inputs_nominal = {
        'outdir': 'results/',
        'case_name': None,
        'spectrum': 'inputs/TOI1266b_spectrum.txt',
        'planet_mass': pl.mass,
        'planet_radius': pl.radius,
        'metallicity': None,
        'x_acc': None,
        'CtoO': None,
        'N_depletion': None,
        'climate_filename': 'results/TOI1266b_MH=2.000_CO=1.000_Tint=45.0.pkl',
        'Kzz': 1e7
    }

    # Escape increases planet b metallicity
    inputs = deepcopy(inputs_nominal)
    inputs['case_name'] = 'TOI1266b_Model1'
    inputs['metallicity'] = 200
    model_utils.run_photochem(**inputs)

    # Magma ocean interaction decreases planet c C/O ratio, and depletes NH3
    inputs = deepcopy(inputs_nominal)
    inputs['case_name'] = 'TOI1266b_Model2'
    inputs['metallicity'] = 50
    inputs['N_depletion'] = 1000
    inputs['CtoO'] = 1
    model_utils.run_photochem(**inputs)

    # Ice accretion makes both planets H2O-rich like GJ 9827 d
    inputs = deepcopy(inputs_nominal)
    inputs['case_name'] = 'TOI1266b_Model3'
    inputs['metallicity'] = 10
    inputs['x_acc'] = 0.5
    model_utils.run_photochem(**inputs)

def make_picasos(filename=None):
    if filename is None:
        filename_db = os.path.join(os.environ['picaso_refdata'],'opacities/')+'opacities.db'
    pl = planets.TOI1266c
    st = planets.TOI1266
    star_kwargs = {
        'temp': st.Teff,
        'metal': st.metal,
        'logg': st.logg
    }
    opannection_kwargs = {
        'wave_range': [0.5, 5.5]
    }
    p_c = model_utils.Picaso(
        filename_db, pl.mass, pl.radius, st.radius, opannection_kwargs=opannection_kwargs, star_kwargs=star_kwargs
    )
    
    pl = planets.TOI1266b
    st = planets.TOI1266
    # filename_db = '/Users/nicholas/Applications/picaso_data/reference/opacities/opacities.db'
    star_kwargs = {
        'temp': st.Teff,
        'metal': st.metal,
        'logg': st.logg
    }
    p_b = model_utils.Picaso(
        filename_db, pl.mass, pl.radius, st.radius, opannection_kwargs=opannection_kwargs, star_kwargs=star_kwargs
    )
    return p_c, p_b

def make_spectra(p, atm):

    all_species = ['CH4', 'CO', 'CO2', 'H2O', 'NH3']
    excude_mols = {'all': [], 'none': all_species}
    for sp in all_species:
        tmp = deepcopy(all_species)
        tmp.remove(sp)
        excude_mols[sp] = tmp

    spectra = {}
    for exclude in excude_mols:
        wavl, rprs2, df = p.rprs2(atm, R=100, atmosphere_kwargs={'exclude_mol': excude_mols[exclude]})
        spectra[exclude] = rprs2
    spectra['wavl'] = wavl
    
    return spectra

def make_all_spectra(p_c, p_b):

    spectra_c = []
    for i in range(1,4):
        with open(f'results/TOI1266c_Model{i}_picaso.pkl','rb') as f:
            atm = pickle.load(f)
        atm = pd.DataFrame(atm)
        spectra_c.append(make_spectra(p_c, atm))
    
    spectra_b = []
    for i in range(1,4):
        with open(f'results/TOI1266b_Model{i}_picaso.pkl','rb') as f:
            atm = pickle.load(f)
        atm = pd.DataFrame(atm)
        spectra_b.append(make_spectra(p_b, atm))

    return spectra_c, spectra_b

def make_pandexo_res():

    from pandexo.engine import justdoit as pandexo_jdi
    inst = pandexo_jdi.load_mode_dict('NIRISS SOSS')
    inst['configuration']['detector']['subarray'] = 'substrip256'
    insts = [
        ('NIRISS SOSS',inst),
        'NIRSpec G395M'
    ]
    
    np.random.seed(0)
    pl = planets.TOI1266c
    st = planets.TOI1266
    pandexo_res_c = model_utils.synthetic_data_pandexo(
        pl, 
        st, 
        starpath='inputs/TOI1266_picaso.txt', 
        insts=insts,
    )
    
    pl = planets.TOI1266b
    pandexo_res_b = model_utils.synthetic_data_pandexo(
        pl, 
        st, 
        starpath='inputs/TOI1266_picaso.txt', 
        insts=insts,
    )
    return pandexo_res_c, pandexo_res_b, insts

def spectra():

    p_c, p_b = make_picasos()
    spectra_c, spectra_b = make_all_spectra(p_c, p_b)
    pandexo_res_c, pandexo_res_b, insts = make_pandexo_res()

    out = {
        'spectra': {'c': spectra_c, 'b': spectra_b},
        'pandexo': {'c': pandexo_res_c, 'b': pandexo_res_b, 'insts': insts}
    }
    with open('results/spectra.pkl', 'wb') as f:
        pickle.dump(out, f)

if __name__ == '__main__':
    # climate()
    # photochemistry()
    spectra()