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
s = ['dog', 'cat', 'lion']
print(s)

# %%
numberOfAnimals = ('dog', 4), ('cat', 7), ('lion', 69)
d, c, l = numberOfAnimals
print(d, c, l)

# %%
uniqueFinder = set('Banana')

# %%
print(uniqueFinder)

# %%
import numpy as np

# %%
x = np.array([1, 4, 3])
x

# %%
y = np.array([[1, 4, 3], [9, 2, 7]])
y

# %%
y.shape

# %%
y.size

# %%
z = np.arange(1, 2000, 1)
z

# %%
a = np.arange(0.5, 3, 0.5)
a

# %%
import numpy as np


# %%
y = np.array([[1, 4, 3], [9, 2, 7]])
y[0, 1]

# %%
y[0, 1]

# %%
y[0]

# %%
y[0, :]

# %%
y[ :, -1]

# %%
y[
