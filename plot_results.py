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
datadir, nobs = "all_edr3_binflag/", None
#datadir, nobs = "all_edr3_upper3/", None
datadir, nobs, ylim = "sim-lognorm_1_0p2/", 50, (0, 2.8)
#datadir, nobs, ylim = "sim-2lognorm_1p1_0p1_0p6_0p1/", 50, (0, 5.7)
#datadir, nobs, ylim = "sim-2lognorm_1p1_0p1_0p6_0p1/", 50*3*3, (0, 4.8)
#datadir, nobs, ylim = "sim-loguni_m0p5_1/", 50*3*3, (0, 5.)
#datadir, nobs = "all_b16macro/", None

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
"""
plike_files = glob.glob(datadir+"*plike.txt")
plikes = []
for i in range(nobs):
    fname = plike_files[i]
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
len(plikes)
"""

#%%
dlogp = np.diff(logpgrid)[0]
plt.figure(figsize=(16*0.7,7*0.7))
plt.xlabel("$\log_{10}(P_\mathrm{rot}/\mathrm{day})$")
plt.ylabel("$\mathrm{probability\ density}$")
plt.xlim(pmin, pmax)
for i in range(len(plikes)):
	y = plikes[i]
	yn = y/np.sum(y*dlogp)
	#if i%4!=0:
	#	continue
	plt.plot(logpgrid, yn, color='gray', lw=0.3)
#plt.savefig(datadir+"posteriors.png", dpi=200, bbox_inches="tight")

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
runname = 'run5'
#runname = 'run6-p09'

#%%
posts = glob.glob(datadir+"gpn100*n%d*%s*.pkl"%(nobs,runname))
#posts = glob.glob(datadir+"*noprior*.pkl")
if not len(posts):
	posts = glob.glob(datadir+"gpn100*%s*.pkl"%runname)
filebase = posts[0].replace("/", "_").split(".")[0]
print (filebase)

#%%
with open(posts[0], 'rb') as f:
    mcmc = dill.load(f)

#%%
priors = np.array(mcmc.get_samples()["priors"])
q = np.percentile(priors, [16, 50, 84], axis=0)
q2 = np.percentile(priors, [5, 95], axis=0)
qmean = np.mean(priors, axis=0)
qstd = np.std(priors, axis=0)

#%%
if datadir=='all_edr3_binflag/':
    qmean_all = qmean

#%%
#plt.hist(np.array(mcmc.get_samples()["norm"]));

#%%
lnpmeans, lnpstds = [], []
for i in range(len(priors)):
	pi = priors[i]
	_mean = np.average(logpgrid, weights=pi)
	lnpmeans.append(_mean)
	lnpstds.append(np.sqrt(np.average((logpgrid-_mean)**2, weights=pi)))
lnpmeans = np.array(lnpmeans)
lnpstds = np.array(lnpstds)

#%%
"""
lnpmeans, lnpvars = [], []
for i in range(len(priors)):
	pi = priors[i]
	_mean = np.average(logpgrid, weights=pi)
	lnpmeans.append(_mean)
	lnpvars.append(np.average((logpgrid-_mean)**2, weights=pi))
lnpmeans = np.array(lnpmeans)
lnpvars = np.array(lnpvars)
lnpstds = np.sqrt(np.mean(lnpvars))
"""

#%%
print ("mean: %.2f +/- %.2f"%(np.mean(lnpmeans), np.std(lnpmeans)))
print ("std: %.2f +/- %.2f"%(np.mean(lnpstds), np.std(lnpstds)))

#%%
"""
N = 1000000
s = np.random.randn(N)*0.1 + 1.1
irep = np.random.rand(N)<0.25
s[irep] = np.random.randn(np.sum(irep))*0.1 + 0.6
print (np.mean(s), np.std(s))
"""

#%%
plt.figure(figsize=(16*factor,7*factor))
plt.xlabel("$\log_{10}P_\mathrm{rot}$ $\mathrm{(days)}$")
plt.ylabel("$\mathrm{probability\ density}$")
plt.xlim(pmin, pmax)
#plt.yscale("log")
if 'sim' in datadir:
    plt.ylim(ylim)
plt.plot(logpgrid, qmean, color="C0", label=r"recovered (mean \& SD)")
plt.fill_between(logpgrid, qmean-qstd, qmean+qstd, color="C0", alpha=0.1)
#plt.fill_between(logpgrid, q2[0], q2[1], color="C0", alpha=0.05)
if 'all' not in datadir:
    plt.plot(logpgrid, truth, '-', color='C1', lw=2, label='truth', ls='dotted')
if 'all' not in datadir:
    plt.plot(logpgrid, plikes_mean/np.sum(plikes_mean*np.diff(logpgrid)[0]), color='gray', lw=3, alpha=0.4, label='mean of individual posteriors')
    plt.hist(pml, density=True, bins=20, histtype='step', lw=1.5, ls='dashed', label='peaks of individual posteriors', color='gray', alpha=0.4)
if 'upper3' in datadir or 'b16macro' in datadir:
    plt.plot(logpgrid, qmean_all, color='C1', ls='dashed', label='mean prediction in Figure 5', lw=1)
plt.legend(loc="best")
plt.title("$%d$ $\mathrm{stars}$"%nobs, fontsize=24)
plt.savefig("plots/"+filebase+"_models.png", dpi=200, bbox_inches="tight")
plt.show()

#%%
def sample_vsini(d, mcmc, N, log=False):
    vmax = np.max(d.vsini)*1.2
    samples = []
    if log:
        bins = np.logspace(0, np.log10(vmax), 20)
        bins_cum = np.logspace(0, np.log10(vmax), 1000)
    else:
        bins = np.linspace(0, vmax, 20)
        bins_cum = np.linspace(0, vmax, 1000)
    vsinis, vsinis_cum = [], []
    logpgrid = mcmc.get_samples()["logpgrid"][0]
    for i in range(N):
        idx = np.random.choice(len(mcmc.get_samples()["priors"]))
        prior = mcmc.get_samples()["priors"][idx]
        priorcum = np.cumsum(prior)/np.cumsum(prior)[-1]
        rnds = np.random.rand(len(d))
        try:
            logpsample = logpgrid[np.digitize(rnds, priorcum)] - 0.5*np.diff(logpgrid)[0]
        except:
            pass
        radsample = d.rad + np.random.randn(len(d))*d.rade
        sinisample = np.sqrt(1-np.random.rand(len(d))**2)
        vsinisample = vunit * radsample / 10**logpsample * sinisample
        vsinisample += np.random.randn(len(d))
        vsinisample[vsinisample<0] = 1e-4
        h, b = np.histogram(vsinisample, bins=bins, density=True)
        vsinis.append(h)
        h, b = np.histogram(vsinisample, bins=bins_cum, density=True)
        vsinis_cum.append(np.cumsum(h)/np.cumsum(h)[-1])
        samples.append(list(vsinisample))
    return bins, np.array(vsinis), bins_cum, np.array(vsinis_cum), np.array(samples)

#%%
bins, h, bins_cum, hc, _s = sample_vsini(dcat, mcmc, 1000, log=True*0)

#%%
if 'all' in filebase:
    dlabel = 'data'
else:
    dlabel = 'simulated data'

#%%
sigv = 0
q = np.percentile(hc, [5, 16, 50, 84, 95], axis=0)
_qmean, _qstd = np.mean(hc, axis=0), np.std(hc, axis=0)
plt.figure(figsize=(10*factor,7*factor))
plt.xscale("log")
#plt.yscale("log")
plt.ylim(0, 1.1)
plt.xlim(bins_cum[0], bins_cum[-1]*0.99)
plt.xlabel("$v\sin i$ $(\mathrm{km/s})$")
plt.ylabel("$\mathrm{cumulative\ distribution\ function}$")
plt.plot(b, np.repeat(_qmean, 2), '-', color='C0', lw=2, alpha=0.4, label=u"prediction (mean \& SD)")
plt.fill_between(b, np.repeat(_qmean-_qstd, 2), np.repeat(_qmean+_qstd, 2), color="C0", alpha=0.1)
if 'upper3' in filebase:
    udata = np.array(dcat.vsini)
    udata[udata<3.] = 3
    udata = np.sort(udata)
    plt.fill_between(np.linspace(bins_cum[0], 3, 100), 0, 12./len(dcat), color='C1', alpha=0.4)
    x, y = np.repeat(udata, 2), np.repeat(np.cumsum(np.ones_like(udata)), 2)/len(dcat)
    plt.plot(x[x>=3][22:], y[x>=3][22:], color='C1', ls='dashed', lw=1.5, label=dlabel)
else:
    udata = np.sort(np.sqrt(dcat.vsini**2-sigv**2))
    x, y = np.repeat(udata, 2), np.repeat(np.cumsum(np.ones_like(udata)), 2)/len(dcat)
    #plt.hist(udata, bins=bins_cum, histtype='step', cumulative=True, density=True, color='C1', ls='dashed', lw=1.5, label="data")
    plt.plot(np.r_[[np.min(x)],x], np.r_[[0],y], color='C1', ls='dashed', lw=1.5, label=dlabel)
b = np.r_[bins_cum[0], np.repeat(bins_cum[1:-1], 2), bins_cum[-1]]
#plt.plot(b, np.repeat(q[2], 2), '-', color='C0', lw=2, alpha=0.4, label=u"prediction\n(mean \% SD)")
#plt.fill_between(b, np.repeat(q[1], 2), np.repeat(q[3], 2), color="C0", alpha=0.1)
#plt.fill_between(b, np.repeat(q[0], 2), np.repeat(q[4], 2), color="C0", alpha=0.1)
plt.legend(loc="best")
plt.title("$%d$ $\mathrm{stars}$"%nobs, fontsize=24)
plt.savefig("plots/"+filebase+"-vsini.png", dpi=200, bbox_inches="tight")
