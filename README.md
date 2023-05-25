# Optimal sizing and placement of Electrical Vehicle charging stations to serve Battery Electric Trucks 

### Master Thesis 2023
### Candidate: Ole-Andre Fagerli Isaksen
### Supervisors: Chiara Bordin and Sambeet Mishra

## Instruction for using the musk-model
To use the musk-model, download shapefiles from: 

https://doi.org/10.5281/zenodo.7968333

They must be places in the folder "*musk/NorthernNorway/Roads/*".

The model is inspired by the github user "obedsims", which can be found at https://github.com/obedsims/Musk-Model.

## To reproduce the networks created in PyPSA-eur do the following
Visit https://pypsa-eur.readthedocs.io/en/latest/installation.html, to setup PyPSA-eur:

When PyPSA-eur is set up, copy the config file from *Masters-Thesis/pypsa-eur/config/config.yaml*.
The environment which I used can be found in *Masters-Thesis/pypsa-eur/environment/*.

To create the network containing the chargers run: 

*snakemake -call resources/networks/elec.nc*

#### To install the chargers to the network use the jupyter-notebook file "Install chargers.ipynb"
It should be placed inside the main folder in the pypsa-eur folder downloaded from *https://github.com/PyPSA/pypsa-eur*, with the csv file *tot_chargers.csv*, found in *Masters-Thesis/pypsa-eur/*.
The csv file contains x, y position in decimal degrees and charger capacity in kW, this is transformed to MW which is the standard unit for PyPSA-eur. It is important to be patient, because this can take some time (approximately 10-15 minutes). Considering that 26 559 chargers are installed to the network file. After the chargers are installed, export the network and replace it with the existing *elec.nc* network in *pypsa-eur/resources/networks*.

After the chargers is succesfully installed to the network, the network can be solved, run:

*snakemake -call results/networks/elec_s_60_ec_lcopt_24H.nc*

The network that is compared can be created by running the last line, but without using jupyter-notebook to install the chargers. It is important to do this in a "clean" pypsa-eur installation, by pulling pypsa-eur once again from github. But remember to use the same config file found in *Masters-Thesis/pypsa-eur/config/*.

### The already solved networks from the thesis can be analysed in the folder *pypsa-eur/Electricity Studies* by using jupyter-notebook
