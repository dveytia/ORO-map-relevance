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
python3 -m pip install scikit-learn
python3 -m pip install transformers
python3 -m pip install tensorflow
python3 -m pip install tensorflow_addons
python3 -m pip install matplotlib
conda install -c conda-forge mpi4py

conda list -n distilBERT_env_3.8 # check

ipython kernel install --user --name=distilBERT_env # ipykernel for running environment in jupyter
```

## Instructions to run the analysis scripts

### Running from jupyter lab

From here, if jupyter-lab is not already open, can run `jupyter-lab` in the terminal, or using launcher open new python console. When opening, there will be a dropdown menu to select "distilBERT_env" (name of the environment specified in ipython kernel install) as to run the kernel


### Running from comand line

Optional: set up screen
```
screen -S [screen_name] # optional to create a screen
conda activate distilBERT_env_3.8
cd ORO-map-relevance

```
The execute script command depending on the label (single or multi) and whether it's selection or predictions

```
## FOR THE BINARY SCREENER (lots of data)
mpiexec -n 5 python model_selection_excl.py 
mpiexec -n 10 python binary_predictions_excl.py # for model selection

## FOR LABELS AND MULTI LABELS
mpiexec -n 3 python [path to script].py # model selection
mpiexec -n 5 python [path to script].py # predictions

```
Optional: to detach from screen Ctrl+A+D
