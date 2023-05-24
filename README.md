# Masters-Thesis
Optimal locations for BETs with grid impacts.

## Instruction for using the musk-model
To use the musk-model, download shapefiles from: 

https://doi.org/10.5281/zenodo.7968333

They must be places in the folder "*musk/NorthernNorway/Roads/*".

The model is inspired by the github user "obedsims", which can be found at https://github.com/obedsims/Musk-Model.

## To reproduce the networks created in PyPSA-eur do the following
Visit https://pypsa-eur.readthedocs.io/en/latest/installation.html, to setup PyPSA-eur:

When PyPSA-eur is set up, copy the config file from *pypsa-eur/config/config.yaml*.
The environment which I used can be found in *pypsa-eur/environment/*.

To create the network containing the chargers run: 

*snakemake -call resources/networks/elec.nc*

To install the chargers use the jupyter-notebook file "Install chargers.ipynb"
After the chargers is succesfully installed to the network, the network can be solved, run:

*snakemake -call results/networks/elec_s_60_ec_lcopt_24H.nc*

The network that is compared can be created by running the last line, but without using jupyter-notebook to install the chargers.

### The networks from the thesis can be analysed in the folder Electricity Studies
