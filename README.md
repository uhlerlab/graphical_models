[![PyPI version](https://badge.fury.io/py/graphical_models.svg)](https://badge.fury.io/py/graphical_models)
[![Build Status](https://travis-ci.com/uhlerlab/graphical_models.svg?branch=main)](https://travis-ci.com/uhlerlab/graphical_models)
[![codecov](https://codecov.io/gh/uhlerlab/graphical_models/branch/main/graph/badge.svg?token=LF0YVTL3GO)](https://codecov.io/gh/uhlerlab/graphical_models)

`graphical_models` is a Python package for:
* **representing** graphical models, including directed acyclic graphs (DAGs), undirected graphs,
(maximal) ancestral graphs (MAGs), partially directed acyclic graphs (PDAGs), partial ancestral graphs (PAGs),

* **generating** graphical models at random, and

* **sampling** from graphical models with specified distributions, e.g. Gaussian DAGs and Gaussian Graphical Models (GGMs).

`graphical_models` is a part of the [causaldag](https://github.com/uhlerlab/causaldag) ecosystem of packages.

### Install
Install the latest version of `graphical_models`:
```
$ pip3 install graphical_models
```

### Documentation
Documentation is available at https://graphical-models.readthedocs.io/en/latest/


### Simple Example

```
>>> from graphical_models import DAG
>>> d = DAG(arcs={(0, 1), (2, 1)})
>>> d.vstructures()
{(0, 1, 2)}
>>> d.cpdag().arcs
{(0, 1), (2, 1)}
>>> d2 = DAG(arcs={(0, 1), (1, 2), (0, 2)})
>>> d2.is_imap(d)
True
>>> d2.markov_equivalent(d)
False
>>> d.dsep(0, 2)
True
>>> d.dsep(0, 2, {1})
False
>>> m = d.moral_graph()
>>> m.edges
{frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}
>>> d3 = DAG(arcs={(0, 1), (0, 2)})
>>> mag = d3.marginal_mag(0)
>>> mag.bidirected
{frozenset({1, 2})}
```

### License

Released under the 3-Clause BSD license (see LICENSE.txt):
```
Copyright (C) 2021
Chandler Squires <csquires@mit.edu>
```
