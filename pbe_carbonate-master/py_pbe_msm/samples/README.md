[![Build Status](https://travis-ci.org/caiofcm/tests2nb.svg?branch=master)](https://travis-ci.org/caiofcm/tests2nb)

# py_pbe_msm: Python building blocks for PBE solution with the Moving Sectional Method

Provides building blocks for the solution of Population Balance Equations using the Moving Sectional Method (MSM) with processes of nucleation, aggregation, growth and dissolution. Multiples particle size distribution are allowed. The code is optimized using the `numba` package for just in time compilation.



## Install

<!-- Install using `pip`:

```bash
pip install git+https://github.com/caiofcm/tests2nb.git
```

Alternatively, you can create a local clone of this repository and install
from it:

```bash
git clone https://github.com/caiofcm/tests2nb.git
pip install -r requirements.txt
``` -->

## Usage


<!-- To convert a python test script into a notebook:

```bash
python -m tests2nb ../samples/test_wallet.py out_.ipynb
``` -->

## Samples

See `samples` directory for examples with different particle processes.


## TODO:

- Better way to pass the aggregation function to the rhs function
    - See: [cfunc](https://numba.pydata.org/numba-doc/dev/user/cfunc.html)
