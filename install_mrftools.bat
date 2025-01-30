@echo off
call conda.bat activate
conda env create --name %1 --file environment.yml
conda activate %1

cd ..
git clone https://github.com/pehses/twixtools.git
cd twixtools
pip install .

cd ..
git clone https://github.com/py-baudin/epgpy.git
pip install epgpy/.
cd epgpy
pip install .