# BART Playground

This is meant to be a modular implementation of BART that allows for easier experimentation.



If any requirements are added, one should add these to the `requirements.txt`. 
This can also be run automatically with 

``
pipreqs . --force
``
In order to install the package as a local package, run:

``
pip install -e .
``
The installation should parse `requirements.txt` and install them first. This will not install dev requirements like `pytest`.