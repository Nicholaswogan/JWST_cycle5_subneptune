# Environment

```sh
conda create -n cycle5 -c conda-forge -c bokeh photochem=0.6.7 numpy=1.24 mpi4py dill tqdm astropy=6.1 matplotlib jupyter uncertainties adjustText pandas pip xarray pathos bokeh=2.4.3 joblib photutils pysynphot sphinx bibtexparser netcdf4 h5netcdf wget unzip tar pymultinest=2.12 scipy=1.11
conda activate cycle5

# Install PICASO
wget https://github.com/natashabatalha/picaso/archive/4d5eded20c38d5e0189d49f643518a7b336a5768.zip
unzip 4d5eded20c38d5e0189d49f643518a7b336a5768.zip
cd picaso-4d5eded20c38d5e0189d49f643518a7b336a5768
python -m pip install . -v
cd ../
cp -r picaso-4d5eded20c38d5e0189d49f643518a7b336a5768/reference picasofiles/
rm -rf picaso-4d5eded20c38d5e0189d49f643518a7b336a5768
rm 4d5eded20c38d5e0189d49f643518a7b336a5768.zip

python -m pip install pandexo.engine==2.0

export picaso_refdata=$(pwd)"/picasofiles/reference/"
# Assume star information is on system
```

# Activate

```sh
conda activate cycle5
export picaso_refdata=$(pwd)"/picasofiles/reference/"
```