#%% Packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import R_jup, R_sun, R_earth, au, M_sun
import pandas as pd
import os, glob, sys, time
import corner
import dill
from scipy.stats import norm
vunit = (2*np.pi*R_sun/86400).value*1e-3

#%%
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='sans-serif')
#sns.set_palette('colorblind')
plt.rcParams["figure.figsize"] = (18,6)
from matplotlib import rc
rc('text', usetex=True)

#%%
#datadir, nobs = "all_edr3_binflag/", None
datadir, nobs = "all_edr3_upper3/", None
#datadir, nobs = "all_d14macro/", None
#datadir, nobs = "all_b16macro/", None

#%%
fname = glob.glob(datadir+"*_rad.csv")[0]
print (fname)

#%%
dcat = pd.read_csv(fname)
if "kepid" not in dcat.keys():
    dcat["kepid"] = dcat.KIC
if "rade" not in dcat.keys():
    dcat["rade"] = 0.5*(dcat.radep + dcat.radem)
print ("Catalog loaded: %s"%fname)

#%% isochrone logg
dciso = pd.merge(dcat, pd.read_csv("isochrone/loudeniso_m14_merged.csv")[['kepid', 'logg_iso']], on='kepid')
#idx = (~dciso.binflag) & (dciso.logg_iso>3.9)
dciso['logg'] = dciso.logg_iso
#dciso.to_csv("all_edr3_upper3_isog/"+fname.split("/")[-1])

#%%
idx = (~dcat.binflag) & (dcat.logg>3.9)
dcat = dcat[idx].reset_index(drop=True)

#%%
dzams = pd.read_csv("isochrone/MIST_1e8yr.txt", delim_whitespace=True, comment='#')
dzams['teff'] = 10**dzams.log_Teff
plt.xlim(3500, 7500)
plt.ylim(0.5, 1.5)
plt.ylabel("mass $M_\odot$")
plt.xlabel("ZAMS $T_\mathrm{eff}$ (K)")
plt.plot(dzams.teff, dzams.initial_mass,  '.')

#%%
print (dzams.initial_mass[np.argmin(np.abs(dzams.teff-5900))], dzams.initial_mass[np.argmin(np.abs(dzams.teff-6200))])

#%%
dciso = pd.merge(dcat, pd.read_csv("isochrone/loudeniso_m14_merged.csv")[['kepid', 'age', 'mass']], on='kepid')
#idxlowt = (5900 < dciso.teff) & (dciso.teff < 6200)
idxlowm = (1.05 < dciso.mass) & (dciso.mass < 1.15)
print (np.sum(idxlowm))
print (np.median(dciso[idxlowm].age))
agecut = 4.3
np.min(dciso[idxlowm & (dciso.age > agecut)].age)

#%%
dciso[idxlowm & (dciso.age < agecut)].reset_index(drop=True).to_csv("all_edr3_upper3_lowmyoung/"+fname.split("/")[-1])
dciso[idxlowm & (dciso.age > agecut)].reset_index(drop=True).to_csv("all_edr3_upper3_lowmold/"+fname.split("/")[-1])

#%%
plt.hist(dciso[idxlowm].age)

#%%
"""
np.mean(dciso[idxlowt & (dciso.age < agecut)].age), np.std(dciso[idxlowt & (dciso.age < agecut)].age)
np.mean(dciso[idxlowt & (dciso.age > agecut)].age), np.std(dciso[idxlowt & (dciso.age > agecut)].age)
dciso[idxlowt & (dciso.age < agecut)].reset_index(drop=True).to_csv("all_edr3_upper3_lowtyoung/"+fname.split("/")[-1])
dciso[idxlowt & (dciso.age > agecut)].reset_index(drop=True).to_csv("all_edr3_upper3_lowtold/"+fname.split("/")[-1])
dciso[idxlowt & (dciso.age > 4.65)].reset_index(drop=True).to_csv("all_edr3_upper3_lowtoldest/"+fname.split("/")[-1])
"""

#%%
tpcts = np.percentile(dcat.teff, [0, 33.3, 66.6, 100])
print (tpcts)

np.min(dcat.teff)
np.max(dcat.teff)
np.sum(dcat.teff<6103.04)
np.sum((6103<dcat.teff)&(dcat.teff<6245.96))
np.sum((dcat.teff>6245.96))
48*3

#%%
for i in range(len(tpcts)-1):
    outdir = datadir[:-1] + "_t%d/"%i
    outname = outdir + fname.split("/")[-1]
    if not os.path.exists(outdir):
        os.system("mkdir %s"%outdir)
    _idx = (tpcts[i]<dcat.teff) & (dcat.teff<tpcts[i+1])
    print (np.min(dcat[_idx].teff)-np.median(dcat[_idx].teff))
    #dcat[_idx].reset_index(drop=True).to_csv(outname, index=False)
