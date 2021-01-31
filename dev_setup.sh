#!/usr/bin/env

python3 -m venv venv
source venv/bin/activate
pip3 install networkx ipython numpy scipy ipdb tqdm twine wheel
pip3 install jedi==0.17.2