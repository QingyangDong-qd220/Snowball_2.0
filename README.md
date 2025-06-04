# Snowball_2.0
Snowball parser version 2.0 for ChemDataExtractor

For installation, simply copy everything in the relex folder and paste them to the relex folder where ChemDataExtractor is instsalled. 

Labelled phrases trained from band-gap sentences are provided in the BandgapTemp_list.pkl file in the checkpoints folder, which can be imported into any Snowball models. For details, please refer to the manual. 

Known bug: there must not be any spaces in "raw_units". 
Hotfix: remove the spaces in the units before passing the text to the Snowball model. 
[![DOI](https://zenodo.org/badge/694649614.svg)](https://doi.org/10.5281/zenodo.15589129)
