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

#import jax
#numpyro.__version__

#%%
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro import set_platform
set_platform('cpu')

#%%
from jax.config import config
config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import device_put

#%%
pmin, pmax = 0., 2.0
#pmin, pmax = -0.5, 2.5

#%%
fname = glob.glob("data/sample_stars.csv")[0]
print (fname)

#%%
dcat = pd.read_csv(fname)
if "kepid" not in dcat.keys():
    dcat["kepid"] = dcat.KIC
if "rade" not in dcat.keys():
    dcat["rade"] = 0.5*(dcat.radep + dcat.radem)
print ("Catalog loaded: %s"%fname)
dcat.vsini[dcat.vsini!=dcat.vsini] = 0

#%% choose likelihood
datadir, nobs = "period_likelihood/", None
datadir, nobs = "period_likelihood_upper3/", None

#%%
if nobs is not None:
    dcat = dcat.iloc[:nobs].reset_index(drop=True)

#%%
if "binflag" in dcat.keys():
    idx = (~dcat.binflag) & (dcat.logg>3.9)
    dcat = dcat[idx].reset_index(drop=True)

#%%
outdir = datadir[:-1] + "_results/"

#%%
print (len(dcat))
print (datadir, outdir)

#%%
if not os.path.exists(outdir):
    os.system("mkdir "+outdir)

#%%
plikes = []
for i in range(len(dcat)):
    fname = datadir + str(int(dcat.iloc[i].kepid)).zfill(9)+"_plike.txt"
    _d = pd.read_csv(fname, delim_whitespace=True, names=["p", "like"])
    if np.sum(_d.like!=_d.like):
        print ("%s includes nan."%fname)
    else:
        _d["like_prob"] = _d.like * np.median(np.diff(np.log10(_d.p)))
        _d["log10p"] = np.log10(_d.p)
        idx = (pmin < _d.log10p)&(_d.log10p < pmax)
        plikes.append(_d[idx].reset_index(drop=True))
print (len(plikes))

#%%
Nbin = 100
modelname, bsigma = "gpn", 10.
null = True*0
noprior = True*0

#%%
logpgrid = jnp.array(jnp.log10(jnp.array(plikes[0].p)))
pbins_prior = jnp.linspace(pmin, pmax+1e-4, Nbin+1)
dbin = pbins_prior[1] - pbins_prior[0]
logpidx = jnp.digitize(logpgrid, pbins_prior) - 1
alpha_max = jnp.log(1./jnp.median(jnp.diff(pbins_prior)))
alpha_mean = jnp.log(1./(pmax-pmin))
pbins_center = 0.5*(pbins_prior[1:] + pbins_prior[:-1])

#%% truths
truth = None

#%%
def design_matrix(plikes):
    nsys = len(plikes)
    X = np.zeros((nsys, len(plikes[0])))
    for i in range(nsys):
        X[i,:] = plikes[i].like_prob
    return X
X = device_put(design_matrix(plikes))
print (np.shape(X))

#%%
import celerite2.jax
from celerite2.jax import terms as jax_terms

def model_gp(X, null=False, noprior=False):
    # priors on hyperparameters
    lna = numpyro.sample("lna", dist.Uniform(low=-5, high=5))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-2, high=2))

    mean = alpha_mean * jnp.ones(Nbin)
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
    gp.compute(pbins_center)

    alphas = numpyro.sample("alphas", dist.Normal(scale=bsigma*jnp.ones(Nbin)))
    norm = jnp.sum(jnp.exp(alphas)) * dbin
    alphas -= jnp.log(norm)
    priors = jnp.exp(alphas[logpidx])
    numpyro.deterministic("norm", jnp.sum(jnp.exp(alphas)*dbin))
    numpyro.deterministic("priors", priors)
    numpyro.deterministic("logpgrid", logpgrid)
    gploglike = gp.log_likelihood(alphas)

    likes = jnp.dot(X, priors)
    loglikelihood = jnp.sum(jnp.log(likes))
    if null:
        loglikelihood *= 0.
    if noprior:
        gploglike *= 0.
    numpyro.factor("loglike", loglikelihood+gploglike)

#%%
target_accept_prob = 0.9
kernel = numpyro.infer.NUTS(model_gp, target_accept_prob=target_accept_prob)

#%%
basename = outdir + "%s%d"%(modelname, Nbin)
if modelname=="gpn":
    basename += "-bs%d"%bsigma
if null:
    basename += "-null"
if noprior:
    basename += "-noprior"
if "sim" in datadir:
	basename += "-n%d"%nobs
print (basename)

#%%
n_sample = 15000
mcmc = numpyro.infer.MCMC(kernel, num_warmup=n_sample, num_samples=n_sample)

#%%
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, X, null=null, noprior=noprior)

#%%
mcmc.print_summary()

#%%
with open(basename+"_mcmc.pkl", "wb") as f:
    dill.dump(mcmc, f)

#%%
"""
with open(basename+"_mcmc.pkl", 'rb') as f:
	mcmc = dill.load(f)
mcmc.print_summary()
"""

#%%
lna = np.array(mcmc.get_samples()["lna"])
lnc = np.array(mcmc.get_samples()["lnc"])
priors = np.array(mcmc.get_samples()["priors"])
q = np.percentile(priors, [16, 50, 84], axis=0)
q2 = np.percentile(priors, [5, 95], axis=0)

#%%
qmean = np.mean(priors, axis=0)
qstd = np.std(priors, axis=0)

#%%
plt.figure()
plt.xlabel("$\log_{10}(P/\mathrm{day})$")# $\mathrm{(days)}$")
plt.ylabel("$\mathrm{probability\ density}$")
plt.xlim(pmin, pmax)
#plt.yscale("log")
#plt.ylim(-0.5, 9.9)
plt.plot(logpgrid, q[1], color="C0", label=u"inferred distribution\n(median \& 68\% \& 90\%)")
plt.fill_between(logpgrid, q[0], q[2], color="C0", alpha=0.1)
plt.fill_between(logpgrid, q2[0], q2[1], color="C0", alpha=0.05)
#plt.plot(logpgrid, qmean, color="C1", label=u"posterior mean")
#plt.fill_between(logpgrid, qmean-qstd, qmean+qstd, color="C1", alpha=0.1)
if not null:
    if truth is not None:
        plt.plot(logpgrid, truth, '-', color='C1', lw=1, ls='dotted', label='input distribution')
    if "sim" in datadir:
        plt.hist(np.log10(dcat.period), density=True, histtype='step', lw=0.6, bins=15, #pbins_prior[::3],
    color='C1', ls='dashed', alpha=1, label='simulated samples (%d)'%len(dcat))
plt.legend(loc="best")
plt.savefig(basename+"_models.png", dpi=200, bbox_inches="tight")
plt.show()

#%%
keys = list(mcmc.get_samples().keys())
keys.remove('alphas')
keys.remove('priors')
keys.remove('logpgrid')
keys.remove('norm')

#%%
if len(keys):
    hyper = pd.DataFrame(data=dict(zip(keys, [mcmc.get_samples()[k] for k in keys])))
    labs = [k.replace("_", "") for k in keys]
    fig = corner.corner(hyper, labels=labs, show_titles="%.2f")
    fig.savefig(basename+"_corner.png", dpi=200, bbox_inches="tight")

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
        h, b = np.histogram(vsinisample, bins=bins, density=True)
        vsinis.append(h)
        h, b = np.histogram(vsinisample, bins=bins_cum, density=True)
        vsinis_cum.append(np.cumsum(h)/np.cumsum(h)[-1])
        samples.append(list(vsinisample))
    return bins, np.array(vsinis), bins_cum, np.array(vsinis_cum), np.array(samples)

#%%
bins, h, bins_cum, hc, _s = sample_vsini(dcat, mcmc, 1000, log=True*0)

#%%
sigv = 0
q = np.percentile(h, [5, 16, 50, 84, 95], axis=0)
plt.figure(figsize=(16,6))
#plt.yscale("log")
plt.xlim(bins[0], bins[-1]*0.99)
plt.xlabel("$v\sin i$ $(\mathrm{km/s})$")
plt.ylabel("$\mathrm{probability\ density}$")
b = np.r_[bins[0], np.repeat(bins[1:-1], 2), bins[-1]]
plt.hist(np.sqrt(dcat.vsini**2-sigv**2), bins=bins, histtype='step', density=True, color='C1', ls='dashed', lw=1, label='data')
plt.plot(b, np.repeat(q[2], 2), '-', color='C0', lw=2, alpha=0.4, label=u"predicted distribution\n(median \& 68\% \& 90\%)")
plt.fill_between(b, np.repeat(q[1], 2), np.repeat(q[3], 2), color="C0", alpha=0.1)
plt.fill_between(b, np.repeat(q[0], 2), np.repeat(q[4], 2), color="C0", alpha=0.1)
plt.legend(loc="best")
plt.savefig(basename+"-vsini.png", dpi=200, bbox_inches="tight")

#%%
sigv = 0
q = np.percentile(hc, [5, 16, 50, 84, 95], axis=0)
plt.figure(figsize=(16,6))
plt.xscale("log")
#plt.yscale("log")
plt.xlim(bins_cum[0], bins_cum[-1]*0.99)
plt.xlabel("$v\sin i$ $(\mathrm{km/s})$")
plt.ylabel("$\mathrm{cumulative\ probability}$")
plt.hist(np.sqrt(dcat.vsini**2-sigv**2), bins=bins_cum, histtype='step', cumulative=True, density=True, color='C1', ls='dashed', lw=1, label="data")
#plt.hist(_s.ravel(), cumulative=True, density=True, bins=bins_cum, histtype='step', color='red')
b = np.r_[bins_cum[0], np.repeat(bins_cum[1:-1], 2), bins_cum[-1]]
plt.plot(b, np.repeat(q[2], 2), '-', color='C0', lw=2, alpha=0.4, label=u"predicted distribution\n(median \& 68\% \& 90\%)")
plt.fill_between(b, np.repeat(q[1], 2), np.repeat(q[3], 2), color="C0", alpha=0.1)
plt.fill_between(b, np.repeat(q[0], 2), np.repeat(q[4], 2), color="C0", alpha=0.1)
plt.legend(loc="best")
plt.savefig(basename+"-vsinicum.png", dpi=200, bbox_inches="tight")
