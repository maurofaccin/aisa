## Auto-Information State Aggregation

This is a python module aimed at partitioning networks through the maximization of Auto-Information.
If you use this code, please cite the following paper:

---

**State aggregations in Markov chains and block models of networks**, <br>
*Faccin, Schaub and Delvenne*,
[ArXiv 2005.00337](https://arxiv.org/abs/2005.00337)

---

The module provides also a function to compute the Entrogram of a network with a suitable partition.
The Entrogram provides a concise, visual characterization of the Markovianity of the dynamics projected to the partition space.
In case you use this, please cite the following paper:

---

> **Entrograms and coarse graining of dynamics on complex networks**, <br>
> *Faccin, Schaub and Delvenne*,
> [Journal of Complex Networks, 6(5) p. 661-678 (2018)](https://academic.oup.com/comnet/article-abstract/6/5/661/4587985), <br>
> [ArXiv 1711.01987](https://arxiv.org/abs/1711.01987)

---

## Getting the code

### Requirements

The following modules are required to `aisa` to work properly:

- `numpy` and `scipy`
- `networkx`
- `tqdm` (optional)

### Install

Download the code [here](https://github.com/maurofaccin/aisa/archive/master.zip) and unzip locally or clone the `git` repository from [Github](https://github.com/maurofaccin/aisa).

On the terminl run:
```
pip install --user path/to/module
```

### Uninstall

On the terminl run:
```
$ pip uninstall aisa
```

## Usage

Read the [online documentation](https://maurofaccin.github.io/aisa).

## License

Copyright: Mauro Faccin (2020)

AISA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AISA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Check LICENSE.txt for details.
