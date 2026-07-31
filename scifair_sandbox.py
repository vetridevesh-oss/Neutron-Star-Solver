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

returns = compare_eos(["EOSBetaFSUGarnet.dat"])

# %%
nl3 = returns["EOSBetaFSUGarnet.dat"]

print(nl3["Mmax"])
print(nl3["R1.4"])
print(nl3["Lambda1.4"])

# %%
