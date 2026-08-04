# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
import time
from neutronstar_solver import compare_eos

returns = compare_eos(["EOSBetaFSUGarnet.dat", "EOSBetaIUFSU.dat", "EOSBetaNL3.dat", "EOSBetaFSUGold2.dat", "EOSBetaRMF022.dat", "EOSBetaRMF028.dat"])

# %%
