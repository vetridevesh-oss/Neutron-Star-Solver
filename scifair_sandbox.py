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

returns = compare_eos(["EOSBetaIUFSU.dat", "EOSBetaFSUGarnet.dat", "EOSBetaRMF022.dat", "EOSBetaTAMUCFSUa.dat", "EOSBetaNL3.dat", "EOSBetaRMF028.dat", "EOSBetaFSUGold2.dat", "EOSBetaTAMUCFSUb.dat", "EOSBetaRMF032.dat", "EOSBetaTAMUCFSUc.dat", "EOSBetaFSU2L47.dat", "EOSBetaFSU2L52.dat", "EOSBetaFSU2L58.dat", "EOSBetaFSU2L63.dat", "EOSBetaFSU2L69.dat", "EOSBetaFSU2L76.dat", "EOSBetaFSU2L85.dat", "EOSBetaFSU2L95.dat", "EOSBetaFSU2L105.dat"])

# %%
