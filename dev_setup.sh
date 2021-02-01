#!/usr/bin/env

python3 -m venv venv
source venv/bin/activate
pip3 install networkx numpy scipy ipdb tqdm
pip3 install ipython twine wheel sphinx_rtd_theme coverage
pip3 install jedi==0.17.2
