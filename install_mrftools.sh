eval "$(conda shell.bash hook)"
conda env create --name $1 --file environment.yml
conda activate $1 

pip install .
