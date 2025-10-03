import warnings
warnings.filterwarnings("ignore")
from photochem.utils import zahnle_rx_and_thermo_files
from photochem.utils import stars
import planets

import stsynphot as sts
import numpy as np



def save_spectrum(outputfile, wv, F, Teff, Teq):
        
    # Tack on a blackbody to extend the spectrum to 100 um
    wv, F = stars.append_blackbody_to_stellar_spectrum(wv, F, Teff, wv_end=100e3, nwb=1000)
        
    # Rescale the spectrum so that it's total bolometric flux matches Teff
    factor = stars.stefan_boltzmann(Teff)/stars.energy_in_spectrum(wv, F)
    F *= factor

    F = stars.scale_spectrum_to_planet(wv, F, Teq=Teq)

    wv, F = stars.rebin_to_needed_resolution(wv, F)

    stars.save_photochem_spectrum(wv, F, outputfile, scale_to_planet=False)

def main():
    # Chemistry
    zahnle_rx_and_thermo_files(
        atoms_names=['H', 'He', 'N', 'O', 'C', 'S'], # We select a subset of the atoms in zahnle_earth.yaml (leave out Cl)
        exclude_species=['S3','S4','S8','S8aer'], # to reduce stiffness
        rxns_filename='inputs/photochem_rxns.yaml',
        thermo_filename='inputs/photochem_thermo.yaml',
        remove_reaction_particles=True # For gas giants, we should always leave out reaction particles.
    )
    # GJ 176
    wv, F = stars.download_muscles_spectrum(
        star_name='GJ176', 
        verbose=True
    )
    # Planet b
    save_spectrum(
        'inputs/TOI1266b_spectrum.txt', 
        wv.copy(), 
        F.copy(), 
        planets.TOI1266.Teff, 
        planets.TOI1266b.Teq
    )
    # Planet c
    save_spectrum(
        'inputs/TOI1266c_spectrum.txt', 
        wv.copy(), 
        F.copy(), 
        planets.TOI1266.Teff, 
        planets.TOI1266c.Teq
    )

    st = planets.TOI1266
    ST_SS = sts.grid_to_spec('ck04models', st.Teff, st.metal, st.logg) 
    wv = ST_SS.waveset.to('um').value
    F = ST_SS(ST_SS.waveset,flux_unit='flam').value
    out = np.empty((2,wv.shape[0]))
    out[0,:] = wv
    out[1,:] = F
    np.savetxt('inputs/TOI1266_picaso.txt', out.T)


if __name__ == '__main__':
    main()





