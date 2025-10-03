# Environment

```sh
mamba env create -f environment.yaml
mamba activate cycle5

# Install picaso
pip install picaso==3.2 -v

# Get reference
wget https://github.com/natashabatalha/picaso/archive/4d907355da9e1dcca36cd053a93ef6112ce08807.zip
unzip 4d907355da9e1dcca36cd053a93ef6112ce08807.zip
cp -r picaso-4d907355da9e1dcca36cd053a93ef6112ce08807/reference picasofiles/
rm -rf picaso-4d907355da9e1dcca36cd053a93ef6112ce08807
rm 4d907355da9e1dcca36cd053a93ef6112ce08807.zip

# Star stuff
wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz
tar -xvzf synphot3.tar.gz
mv grp picasofiles/
rm synphot3.tar.gz

# opacity db
wget https://zenodo.org/records/3759675/files/opacities.db
mv opacities.db picasofiles/reference/opacities/

export picaso_refdata=$(pwd)"/picasofiles/reference/"
export PYSYN_CDBS=$(pwd)"/picasofiles/grp/redcat/trds"

# python -m pip install pandexo.engine==2.0
# export picaso_refdata='/Users/nicholas/Applications/picaso_data/reference/'
```

# Activate

```sh
conda activate cycle5
export picaso_refdata=$(pwd)"/picasofiles/reference/"
```