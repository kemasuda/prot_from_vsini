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
plt.rcParams["figure.figsize"] = (18-2,6+1)
from matplotlib import rc
rc('text', usetex=True)

#%%
datadir, nobs, ylim = "sim-lognorm_1_0p2/", 50, (0, 2.8)
datadir, nobs, ylim = "sim-2lognorm_1p1_0p1_0p6_0p1/", 50, (0, 4.8)
#datadir, nobs, ylim = "sim-loguni_m0p5_1/", 50, (0, 5.)

#%%
pmin, pmax = 0., 2.0
factor = 0.8

#%%
fname = glob.glob(datadir+"*_rad*.csv")[0]
dcat = pd.read_csv(fname)
if "kepid" not in dcat.keys():
    dcat["kepid"] = dcat.KIC
len(dcat)

#%%
if 'rade' not in dcat.keys():
    dcat['rade'] = 0.5*(dcat.radep + dcat.radem)
if 'binflag' in dcat.keys():
    idx = (~dcat.binflag) & (dcat.logg>3.9)
    dcat = dcat[idx].reset_index(drop=True)

#%%
if nobs is None:
	nobs = len(dcat)

#%%
plikes = []
for i in range(nobs):
    fname = datadir + str(int(dcat.iloc[i].kepid)).zfill(9)+"_plike.txt"
    _d = pd.read_csv(fname, delim_whitespace=True, names=["p", "like"])
    _d["like_prob"] = _d.like * np.median(np.diff(np.log10(_d.p)))
    _d["log10p"] = np.log10(_d.p)
    idx = (pmin < _d.log10p)&(_d.log10p < pmax)
    _plike = _d[idx].like_prob
    if np.sum(_plike!=_plike):
	       print (fname)
    plikes.append(list(_plike/np.sum(_plike)))
    logpgrid = np.log10(_d.p[idx])
plikes = np.array(plikes)
logpgrid = np.array(logpgrid)
print (len(plikes))

#%%
if "edr3" in datadir:
    truth = None
elif "sim-lognorm" in datadir:
    truth = norm.pdf(logpgrid, 1, 0.2)
elif "sim-2lognorm_1_0p1" in datadir:
    truth = 0.7*norm.pdf(logpgrid, 1, 0.1) + 0.3*norm.pdf(logpgrid, 0.5, 0.1)
elif "sim-2lognorm_1p1" in datadir:
    truth = 0.75*norm.pdf(logpgrid, 1.1, 0.1) + 0.25*norm.pdf(logpgrid, 0.6, 0.1)
elif "loguni" in datadir:
    truth = np.ones_like(logpgrid)*2
    truth[(logpgrid<0.5)+(logpgrid>1)] = 0.

#%%
outdir = datadir

#%%
plikes_mean = np.mean(plikes[:nobs], axis=0)

#%%
pml = []
for i in range(len(plikes)):
    pml.append(logpgrid[np.argmax(plikes[i])])
pml = np.array(pml)

#%%
#runname = 'run5'
runname = 'run6-p09'

#%%
posts = glob.glob(datadir[:-1]+"_MC/"+"*gpn100*n%d*%s*.pkl"%(nobs,runname))
posts[0]
filebase = datadir[:-1]+"_MC"
print (filebase)

#%%
qmeans = []
for post in posts:
    with open(post, 'rb') as f:
        mcmc = dill.load(f)

    #
    priors = np.array(mcmc.get_samples()["priors"])
    q = np.percentile(priors, [16, 50, 84], axis=0)
    q2 = np.percentile(priors, [5, 95], axis=0)
    qmean = np.mean(priors, axis=0)
    qstd = np.std(priors, axis=0)

    #
    lnpmeans, lnpstds = [], []
    for i in range(len(priors)):
    	pi = priors[i]
    	_mean = np.average(logpgrid, weights=pi)
    	lnpmeans.append(_mean)
    	lnpstds.append(np.sqrt(np.average((logpgrid-_mean)**2, weights=pi)))
    lnpmeans = np.array(lnpmeans)
    lnpstds = np.array(lnpstds)

    #
    #print ("mean: %.2f +/- %.2f"%(np.mean(lnpmeans), np.std(lnpmeans)))
    #print ("std: %.2f +/- %.2f"%(np.mean(lnpstds), np.std(lnpstds)))

    qmeans.append(qmean)

#%%
qmeans = np.array(qmeans)
qmean = np.mean(qmeans, axis=0)
qstd = np.std(qmeans, axis=0)

#%%
plt.figure(figsize=(16*factor,7*factor))
plt.xlabel("$\log_{10}P_\mathrm{rot}$ $\mathrm{(days)}$")
plt.ylabel("$\mathrm{probability\ density}$")
plt.xlim(pmin, pmax)
#plt.yscale("log")
if 'sim' in datadir:
    plt.ylim(ylim)
for _qmean in qmeans:
    plt.plot(logpgrid, _qmean, color="C0", lw=0.2, alpha=0.5)
plt.plot(logpgrid, qmean, color="C0", label=r"recovered (mean \& SD)")
plt.fill_between(logpgrid, qmean-qstd, qmean+qstd, color="C0", alpha=0.1)
#plt.fill_between(logpgrid, q2[0], q2[1], color="C0", alpha=0.05)
if 'all' not in datadir:
    plt.plot(logpgrid, truth, '-', color='C1', lw=2, label='truth', ls='dotted')
plt.legend(loc="best")
plt.title("$%d$ $\mathrm{stars}$"%nobs, fontsize=24)
plt.savefig(filebase+"/results_n%d.png"%nobs, dpi=200, bbox_inches="tight")
plt.show()
