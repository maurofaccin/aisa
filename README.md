## Auto-Information State Aggregation

This is a python module aimed at partitioning networks through the maximization of Auto-Information.
If you use this code, please cite the following paper:

> **State aggregations in Markov chains and block models of networks**, <br>
> *Faccin, Schaub and Delvenne*,
> [Phys. Rev. Lett., 127(7) p.078301 (2021)](https://doi.org/10.1103/PhysRevLett.127.078301)<br>
> [ArXiv 2005.00337](https://arxiv.org/abs/2005.00337)

(data used in the paper to analyse the ocean surface currents can be found in the GitHub repo [`ocean_surface_dataset`](https://github.com/maurofaccin/ocean_surface_dataset))

The module provides also a function to compute the Entrogram of a network with a suitable partition.
The Entrogram provides a concise, visual characterization of the Markovianity of the dynamics projected to the partition space.
In case you use this, please cite the following paper:

> **Entrograms and coarse graining of dynamics on complex networks**, <br>
> *Faccin, Schaub and Delvenne*,
> [Journal of Complex Networks, 6(5) p. 661-678 (2018)](https://academic.oup.com/comnet/article-abstract/6/5/661/4587985), <br>
> [ArXiv 1711.01987](https://arxiv.org/abs/1711.01987)

## Getting the code

### Requirements

`aisa` requires the following modules to work properly:

- `numpy` and `scipy`
- `networkx`
- `tqdm` (optional)

### Install

#### Pip

`AISA` can be installed directly from PyPI using `pip` with the following:
```
pip install aisa
```

#### Manually

Alternatively one can download the code [here](https://github.com/maurofaccin/aisa/archive/main.zip) and unzip locally or clone the `git` repository from [GitHub](https://github.com/maurofaccin/aisa).
From inside the module folder you can run:
```
pip install aisa
```

### Uninstall

On the terminal run:
```
$ pip uninstall aisa
```

## Usage

Read the [online documentation](https://maurofaccin.github.io/aisa) that describes all classes and functions of the module.

Some simple notebook examples on the usage of this module are provided in the `examples` subfolder:

- a simple example of computing and drawing the `entrogram` and detecting the partition that maximize the auto-information in a well-know small social network, see in [nbviewer](https://nbviewer.jupyter.org/github/maurofaccin/aisa/blob/main/examples/Karate_Club.ipynb)
- an example on how to build a *range dependent network* and find the partition that maximize auto-nformation, see in [nbviewer](https://nbviewer.jupyter.org/github/maurofaccin/aisa/blob/main/examples/Range_Dependent_Network.ipynb)

## License

Copyright: Mauro Faccin (2021)

AISA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AISA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Check LICENSE.txt for details.
