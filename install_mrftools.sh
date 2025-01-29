conda env create --name $1 --file environment.yml
conda activate $1 

cd..
git clone https://github.com/pehses/twixtools.git
pip install twixtools/.

git clone https://github.com/py-baudin/epgpy.git
pip install epgpy/.