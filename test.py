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
from test_parallel import EOStoObservables_parallel, compare_eos

out = EOStoObservables_parallel("EOSBetaNL3.dat")

# %%
