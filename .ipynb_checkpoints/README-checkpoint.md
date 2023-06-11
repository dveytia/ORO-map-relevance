# ORO-map-relevance

## Instructions to make

### Create an environment using Conda

From gitbash, open `jupyter-lab`. Then using the launcher, open the Terminal:
```
conda create --name distilBERT_env_3.8 python=3.8
conda activate distilBERT_env_3.8

python3 -m pip install ipykernel
python3 -m pip install pandas
python3 -m pip install numpy
python3 -m pip install sklearn
python3 -m pip install transformers
python3 -m pip install tensorflow
python3 -m pip install tensorflow_addons
conda install -c conda-forge mpi4py

conda list -n distilBERT_env_3.8 # check

ipython kernel install --user --name=distilBERT_env # Then use ipykernel to open the environment in jupyter
```

From here, if jupyter-lab is not already open, can run `jupyter-lab` in the terminal, or using launcher open new python console. When opening, there will be a dropdown menu to select "distilBERT_env" as to run the kernel