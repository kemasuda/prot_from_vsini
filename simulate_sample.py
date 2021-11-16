#%% Packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import R_jup, R_sun, R_earth, au, M_sun
import pandas as pd
import os
vunit = (2*np.pi*R_sun/86400).value*1e-3

#%%
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='sans-serif')
#sns.set_palette('colorblind')
plt.rcParams["figure.figsize"] = (18,6)

#%%
d = pd.read_csv("all_edr3_binflag/all_edr3_binflag_rad.csv")
d["rade"] = 0.5*(d.radep+d.radem)
idx = (~d.binflag) & (d.logg>3.9)
d = d[idx].reset_index(drop=True)

#%%
plt.hist(d.phot_g_mean_mag)

#%%
Nsample = 500

#%%
rads_true = []
for i in np.random.randint(len(d), size=Nsample):
    rads_true.append(d.rad[i]*(1+np.random.randn()*d.rade[i]))
rads_true = np.array(rads_true)
rads_true

#%%
bins = np.linspace(0.9, 4.0, 50)
plt.hist(d.rad, density=True, bins=bins, histtype='step', lw=1)
plt.hist(rads_true, density=True, bins=bins, histtype='step', lw=1)

#%%
#rade_obs = 0.04
#rads_obs = rads_true * (1+np.random.randn(Nsample)*rade_obs)

#%%
rade_obs = 0.04 * rads_true
rads_obs = rads_true + np.random.randn(Nsample) * rade_obs

#%%
np.random.seed(123)
name = "lognorm_1_0p2"
periods = 10**(1.+np.random.randn(Nsample)*0.2)

#%%
"""
np.random.seed(123)
name = "2lognorm_1_0p2_0p5_0p1"
periods = 10**(1 + np.random.randn(Nsample)*0.2)
periods2 = 10**(0.5 + np.random.randn(Nsample)*0.1)
idx2 = np.random.rand(Nsample)<0.3
periods[idx2] = periods2[idx2]

#%%
np.random.seed(123)
name = "2lognorm_1_0p1_0p5_0p1"
periods = 10**(1 + np.random.randn(Nsample)*0.1)
periods2 = 10**(0.5 + np.random.randn(Nsample)*0.1)
idx2 = np.random.rand(Nsample)<0.3
periods[idx2] = periods2[idx2]
"""

#%%
np.random.seed(123)
name = "loguni_m0p5_1"
periods = 10**(0.5 + np.random.rand(Nsample)*0.5)

#%%
np.random.seed(123)
name = "2lognorm_1p1_0p1_0p6_0p1"
periods = 10**(1.1 + np.random.randn(Nsample)*0.1)
periods2 = 10**(0.6 + np.random.randn(Nsample)*0.1)
idx2 = np.random.rand(Nsample)<0.25
periods[idx2] = periods2[idx2]

#%%
plt.xlim(0, 2)
plt.hist(np.log10(periods[:500]), density=True);

#%%
np.random.seed(123)
cosis = np.random.rand(Nsample)
vsinis_true = vunit * rads_true / periods * np.sqrt(1-cosis**2)
vsinis_obs = vsinis_true + np.random.randn(Nsample)
print (np.min(vsinis_true))
print (np.min(vsinis_obs))
vsinis_obs[vsinis_obs<0] = 0

#%%
kics = np.arange(Nsample)
outdir = "sim-" + name
if not os.path.exists(outdir):
    os.system("mkdir %s"%outdir)

#%%
pd.DataFrame(data={"kepid": kics, "rad": rads_obs, "rade": rade_obs, "rad_true": rads_true, "vsini": vsinis_obs, "vsini_true": vsinis_true, "period": periods, "cosi_true": cosis}).to_csv(outdir+"/samples_rad.csv", index=False)
