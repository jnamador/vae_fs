# Purpose

The prupose of this repo is to recreate the DNN VAE model in [arXiv:2108.03986](https://arxiv.org/abs/2108.03986).

This code freely borrows from the [ToyVAE notebooks](https://github.com/Kenny-Jia/ToyVAE/tree/GAN-dev/software_dev/Notebooks) by Kenny Jia in collaboration with Julia Gonski at SLAC as part of
the RENEW-SJSU program.

## requirements.txt

This is run on NERSC's Perlmutter GPU's. If running on NERSC, do not reinstall from requirements.txt.
Simply load tensorflow with `module load tensorflow/2.12.0` and install
ipykernal with `pip install --user ipykernel` as per the [NERSC Documentation](https://docs.nersc.gov/machinelearning/tensorflow/).

## Running notebooks on NERSC

After installing `ipykernel` as above, run `jupyter notebook --no-browser` in the same terminal and run from there.
