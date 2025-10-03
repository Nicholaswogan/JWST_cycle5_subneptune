from main import make_picasos
import numpy as np
import pandas as pd
import model_utils
import pickle
from pymultinest.solve import solve
import os

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

# Model at wavl
def _model_atm(y, p, R=100, nz=60, **kwargs):

    log10CH4, log10CO2, log10CO, log10H2O, log10NH3, T, log10Ptop_cld, offset_soss, offset_g395m = y

    # Compute H2 and He 
    log10f = np.array([log10CH4, log10CO2, log10CO, log10H2O, log10NH3])
    f = 10.0**log10f
    f_H2_He = 1.0 - np.sum(f)
    assert f_H2_He > 0.0
    f_H2 = f_H2_He*0.85
    f_He = f_H2_He*0.15
    f_CH4, f_CO2, f_CO, f_H2O, f_NH3 = f

    # Pressure
    log10Pmax = 2
    P = np.logspace(-7,log10Pmax,nz)

    # Create dict
    atm = {
        'pressure': P,
        'temperature': np.ones(nz)*T,
        'CH4': np.ones(nz)*f_CH4,
        'CO2': np.ones(nz)*f_CO2,
        'CO': np.ones(nz)*f_CO,
        'H2O': np.ones(nz)*f_H2O,
        'NH3': np.ones(nz)*f_NH3,
        'H2': np.ones(nz)*f_H2,
        'He': np.ones(nz)*f_He,
    }
    atm = pd.DataFrame(atm)

    # Clouds.
    p.clouds_reset()
    dlog10Pcloud = log10Pmax - log10Ptop_cld
    assert dlog10Pcloud > 0

    wavl, rprs2, _ = p.rprs2(atm, R=R, log10Pcloudbottom=log10Pmax, dlog10Pcloud=dlog10Pcloud, **kwargs)

    return wavl, rprs2

def model_atm(y, data_dict, p, **kwargs):
    wavl, rprs2 = _model_atm(y, p, **kwargs)

    log10CH4, log10CO2, log10CO, log10H2O, log10NH3, T, log10Ptop_cld, offset_soss, offset_g395m = y

    rprs2_soss = model_utils.interp_spectrum_to_data(data_dict['NIRISS SOSS']['wv_bins'], wavl, rprs2)
    rprs2_soss += offset_soss

    rprs2_g395m = model_utils.interp_spectrum_to_data(data_dict['NIRSpec G395M']['wv_bins'], wavl, rprs2)
    rprs2_g395m += offset_g395m

    rprs2_at_data = np.concatenate((rprs2_soss, rprs2_g395m))

    return rprs2_at_data

def check_implicit_prior_atm(cube):
    log10CH4, log10CO2, log10CO, log10H2O, log10NH3, T, log10Ptop_cld, offset_soss, offset_g395m = cube
    log10f = np.array([log10CH4, log10CO2, log10CO, log10H2O, log10NH3])
    f = 10.0**log10f
    within_implicit_priors = True
    if np.sum(f) > 1.0:
        within_implicit_priors = False
    return within_implicit_priors 

def prior_atm(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], -13, -0.02) # log10CH4
    params[1] = quantile_to_uniform(cube[1], -13, -0.02) # log10CO2
    params[2] = quantile_to_uniform(cube[2], -13, -0.02) # log10CO
    params[3] = quantile_to_uniform(cube[3], -13, -0.02) # log10H2O
    params[4] = quantile_to_uniform(cube[4], -13, -0.02) # log10NH3
    params[5] = quantile_to_uniform(cube[5], 150, 650) # T
    params[6] = quantile_to_uniform(cube[6], -6.0, 1) # log10Ptop_cld
    params[7] = quantile_to_uniform(cube[7], -1000.0e-6, 1000.0e-6) # offset_soss
    params[8] = quantile_to_uniform(cube[8], -1000.0e-6, 1000.0e-6) # offset_g395m
    return params

def make_loglike(model, check_implicit_prior, data_dict, p):
    def loglike(cube):

        y, e = np.zeros((0)), np.zeros((0))
        for key in data_dict:
            y = np.append(y, data_dict[key]['rprs2'])
            e = np.append(e, data_dict[key]['err'])

        within_implicit_priors = check_implicit_prior(cube)

        if within_implicit_priors:
            resulty = model(cube, data_dict, p)
            loglikelihood = -0.5*np.sum((y - resulty)**2/e**2)
        else:
            loglikelihood = -1.0e100
        return loglikelihood
    return loglike

def make_loglike_prior(p, data_dict, model, _model, prior, check_implicit_prior, param_names):

    loglike = make_loglike(model, check_implicit_prior, data_dict, p)

    out = {
        'loglike': loglike,
        'prior': prior,
        'param_names': param_names,
        'data_dict': data_dict,
        'model': model,
        '_model': _model,
        'p': p
    }

    return out

def make_data():

    with open('results/spectra.pkl','rb') as f:
        res = pickle.load(f)
    spectra = res['spectra']
    pandexo = res['pandexo']
    spectra_c, spectra_b = spectra['c'], spectra['b']
    pandexo_res_c, pandexo_res_b, insts = pandexo['c'], pandexo['b'], pandexo['insts']

    data_dicts_c = []
    data_dicts_b = []
    for i in range(3):
        data_dict = model_utils.synthetic_data(
            pandexo_res=pandexo_res_c,
            insts=insts, 
            R=[None,None],
            dwv=[0.05,0.05],
            ntrans=[2,2], 
            inflation=1.15, 
            wavl_true=spectra_c[i]['wavl'], 
            rprs_true=spectra_c[i]['all'], 
            random=False
        )
        data_dicts_c.append(data_dict)

        data_dict = model_utils.synthetic_data(
            pandexo_res=pandexo_res_b,
            insts=insts, 
            R=[None,None],
            dwv=[0.05,0.05],
            ntrans=[1,1], 
            inflation=1.15, 
            wavl_true=spectra_b[i]['wavl'], 
            rprs_true=spectra_b[i]['all'], 
            random=True
        )
        data_dicts_b.append(data_dict)

    return data_dicts_c, data_dicts_b

def make_cases():

    cases = {}

    param_names = [
        'log10CH4', 'log10CO2', 'log10CO', 'log10H2O', 'log10NH3', 
        'T', 'log10Ptop_cld', 'offset_soss', 'offset_g395m'
    ]

    data_dicts_c, data_dicts_b = make_data()

    # Planet c
    cases['c1'] = make_loglike_prior(PICASO_C, data_dicts_c[0], model_atm, _model_atm, prior_atm, check_implicit_prior_atm, param_names)
    cases['c2'] = make_loglike_prior(PICASO_C, data_dicts_c[1], model_atm, _model_atm, prior_atm, check_implicit_prior_atm, param_names)
    cases['c3'] = make_loglike_prior(PICASO_C, data_dicts_c[2], model_atm, _model_atm, prior_atm, check_implicit_prior_atm, param_names)

    # Planet b
    cases['b1'] = make_loglike_prior(PICASO_B, data_dicts_b[0], model_atm, _model_atm, prior_atm, check_implicit_prior_atm, param_names)
    cases['b2'] = make_loglike_prior(PICASO_B, data_dicts_b[1], model_atm, _model_atm, prior_atm, check_implicit_prior_atm, param_names)
    cases['b3'] = make_loglike_prior(PICASO_B, data_dicts_b[2], model_atm, _model_atm, prior_atm, check_implicit_prior_atm, param_names)

    return cases

PICASO_C, PICASO_B = make_picasos()
RETRIEVAL_CASES = make_cases()

if __name__ == '__main__':

    models_to_run = list(RETRIEVAL_CASES.keys())
    for model_name in models_to_run:
        # Setup directories
        outputfiles_basename = f'pymultinest/{model_name}/{model_name}'
        try:
            os.mkdir(f'pymultinest/{model_name}')
        except FileExistsError:
            pass

        # Do nested sampling
        results = solve(
            LogLikelihood=RETRIEVAL_CASES[model_name]['loglike'], 
            Prior=RETRIEVAL_CASES[model_name]['prior'], 
            n_dims=len(RETRIEVAL_CASES[model_name]['param_names']), 
            outputfiles_basename=outputfiles_basename, 
            verbose=True,
            n_live_points=1000
        )
        # Save pickle
        pickle.dump(results, open(outputfiles_basename+'.pkl','wb'))