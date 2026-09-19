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
from neutronstar_solver import compare_eos
results = compare_eos(["EOSBetaIUFSU.dat", "EOSBetaTAMUCFSUa.dat", "EOSBetaNL3.dat"])

# %%
