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
datadir, nobs = "all_edr3_binflag/", None
datadir, nobs = "all_edr3_upper3/", None
datadir, nobs = "all_d14macro/", None
datadir, nobs = "all_b16macro/", None
datadir, nobs = "all_edr3_binflag_t0/", None
datadir, nobs = "all_edr3_binflag_t1/", None
datadir, nobs = "all_edr3_binflag_t2/", None
datadir, nobs = "all_edr3_upper3_t0/", None
datadir, nobs = "all_edr3_upper3_t1/", None
datadir, nobs = "all_edr3_upper3_t2/", None
datadir, nobs = "all_b16macro_t0/", None
datadir, nobs = "all_b16macro_t1/", None
datadir, nobs = "all_b16macro_t2/", None
datadir, nobs = "sim-lognorm_1_0p2/", 50*2
datadir, nobs = "sim-2lognorm_1p1_0p1_0p6_0p1/", 50*2
datadir, nobs = "sim-loguni_m0p5_1/", 50*2
datadir, nobs = "all_edr3_upper3_lowtyoung/", None
datadir, nobs = "all_edr3_upper3_lowtold/", None
datadir, nobs = "all_edr3_upper3_lowtoldest/", None
datadir, nobs = "all_edr3_upper3_lowmyoung/", None
datadir, nobs = "all_edr3_upper3_lowmold/", None
datadir, nobs = "all_edr3_upper3_isog/", None
datadir, nobs = "sim-2lognorm_1p1_0p1_0p6_0p1/", 50*3*3
datadir, nobs = "sim-lognorm_1_0p2/", 50
datadir, nobs = "sim-loguni_m0p5_1/", 50*9

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
dcat.vsini[dcat.vsini!=dcat.vsini] = 0

#%%
if nobs is not None:
    dcat = dcat.iloc[:nobs].reset_index(drop=True)

#%%
if "binflag" in dcat.keys():
    idx = (~dcat.binflag) & (dcat.logg>3.9)
    dcat = dcat[idx].reset_index(drop=True)

#%%
outdir = datadir
if "upper3_" in outdir or "_lowt" in outdir or "_lowm" in outdir:
    #datadir = outdir[:-4] + "/"
    datadir = "_".join(datadir.split("_")[:-1])+"/"

#%%
print (len(dcat))
print (datadir, outdir)

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
modelname, eps_smooth = "smooth", 8.0
modelname, bsigma = "gpn", 10.
#modelname, bsigma = "gpu", 10.
#modelname = "gpdelta"
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
if "control" in datadir:
    truth = None
elif "sim-lognorm" in datadir:
    truth = norm.pdf(logpgrid, 1, 0.2)
elif "sim-2lognorm_1_0p1" in datadir:
    truth = 0.7*norm.pdf(logpgrid, 1, 0.1) + 0.3*norm.pdf(logpgrid, 0.5, 0.1)
elif "sim-2lognorm_1p1_0p1_0p6_0p1" in datadir:
    truth = 0.75*norm.pdf(logpgrid, 1.1, 0.1) + 0.25*norm.pdf(logpgrid, 0.6, 0.1)
elif "2lognorm" in datadir:
    truth = 0.7*norm.pdf(logpgrid, 1, 0.2) + 0.3*norm.pdf(logpgrid, 0.5, 0.1)
elif "loguni" in datadir:
    truth = np.ones_like(logpgrid)*2
    truth[(logpgrid<0.5)+(logpgrid>1)] = 0.

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
def smoothmodel(X, eps=eps_smooth):
    alphas = numpyro.sample("alphas", dist.Uniform(low=-10*jnp.ones(Nbin), high=alpha_max*jnp.ones(Nbin)))
    priors = jnp.exp(alphas[logpidx])
    numpyro.deterministic("priors", priors)
    likes = jnp.dot(X, priors)
    loglikelihood = jnp.sum(jnp.log(likes)) - 0.5*eps*jnp.sum((alphas[:-1]-alphas[1:])**2)
    norm = -1e4*(1.-jnp.sum(jnp.exp(alphas)*dbin))**2
    numpyro.deterministic("norm", jnp.sum(jnp.exp(alphas)*dbin))
    numpyro.factor("loglike", loglikelihood+norm)

def model_gp(X, uniform=False, null=False, noprior=False):
    #lna = numpyro.sample("lna", dist.Uniform(low=-10, high=10))
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-10, high=10))
    # run3
    #lna = numpyro.sample("lna", dist.Uniform(low=-5, high=5))
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=5))
    # run4
    #lna = numpyro.sample("lna", dist.Uniform(low=-5, high=5))
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-2.5, high=2.5))
    # run5
    #lna = numpyro.sample("lna", dist.Uniform(low=-5, high=5))
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-2, high=2))
    # run 6
    lna = numpyro.sample("lna", dist.Uniform(low=-2, high=2))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-1.5, high=1.5))
    #lna = numpyro.sample("lna", dist.Uniform(low=-3, high=3))
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-10, high=10))
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-1.5, high=10)) # lncprior
    #lnc = numpyro.sample("lnc", dist.TruncatedNormal(low=-1.5)) # lncprior2
    #lnc = 0. # lcprior3
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-2, high=2)) # lncprior4; best for Nbin=100
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-3, high=3)) # lncprior5
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-2.5, high=2.5)) # lncprior6
    #lna = numpyro.sample("lna", dist.Uniform(-1, 1))
    #lna = 1
    #lnc = numpyro.sample("lnc", dist.TruncatedNormal(low=-2.5))
    # acprior2
    #lna = numpyro.sample("lna", dist.Uniform(low=-2, high=2))
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-2, high=2))
    # acprior3
    #lna = 1
    #lnc = numpyro.sample("lnc", dist.Uniform(low=-3, high=3))
    # alpha
    #_alpha_mean = numpyro.sample("alpha_mean", dist.Normal(alpha_mean, 5))
    #mean = _alpha_mean * jnp.ones(Nbin)
    mean = alpha_mean * jnp.ones(Nbin)
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
    gp.compute(pbins_center)

    if uniform:
        alphas = numpyro.sample("alphas", dist.Uniform(low=-bsigma*jnp.ones(Nbin), high=bsigma*jnp.ones(Nbin)))
    else:
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

def model_delta(X, uniform=False, null=False, noprior=False):
    lna = numpyro.sample("lna", dist.Uniform(low=-10, high=10))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-10, high=10))
    mean = alpha_mean * jnp.ones(Nbin)
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
    gp.compute(pbins_center)

    alphas = numpyro.sample("alphas", dist.Uniform(low=-bsigma*jnp.ones(Nbin), high=alpha_max*jnp.ones(Nbin)))
    norm = jnp.sum(jnp.exp(alphas)) * dbin
    normloglike = -1e4*(1.-norm)**2
    priors = jnp.exp(alphas[logpidx])
    numpyro.deterministic("norm", norm)
    numpyro.deterministic("priors", priors)
    numpyro.deterministic("logpgrid", logpgrid)
    gploglike = gp.log_likelihood(alphas)

    likes = jnp.dot(X, priors)
    loglikelihood = jnp.sum(jnp.log(likes))
    numpyro.factor("loglike", loglikelihood+gploglike+normloglike)

#%%
target_accept_prob = 0.9

#%%
if modelname=="smooth":
    kernel = numpyro.infer.NUTS(smoothmodel)
elif modelname=="gpn":
    kernel = numpyro.infer.NUTS(model_gp, target_accept_prob=target_accept_prob)
    uniform = False
elif modelname=="gpu":
    kernel = numpyro.infer.NUTS(model_gp)
    uniform = True
elif modelname=="gpdelta":
    kernel = numpyro.infer.NUTS(model_delta)

#%%
basename = outdir + "%s%d"%(modelname, Nbin)
if modelname=="smooth":
    basename += "-eps%.1f"%eps_smooth
if modelname=="gpn" or modelname=="gpu":
    basename += "-bs%d"%bsigma
if null:
    basename += "-null"
if noprior:
    basename += "-noprior"
if "sim" in datadir:
	basename += "-n%d"%nobs

#%%
#basename += "-lncprior5"
#basename += "-acprior2"
#basename += "-acprior3"
#basename += "-run5"
#basename += "-run6"
#basename += "-run5-p09"
basename += "-run6-p09"
print (basename)

#%%
n_sample = 3000
mcmc = numpyro.infer.MCMC(kernel, num_warmup=n_sample, num_samples=n_sample)

#%%
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, X, uniform=uniform, null=null, noprior=noprior)

#%%
mcmc.print_summary()

#%%
with open(basename+"_mcmc.pkl", "wb") as f:
    dill.dump(mcmc, f)
basename

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
    #if truth is not None:
    #    plt.plot(logpgrid, truth, '-', color='C1', lw=1, ls='dotted', label='input distribution')
    if "sim" in datadir:
        plt.hist(np.log10(dcat.period), density=True, histtype='step', lw=0.6, bins=15, #pbins_prior[::3],
    color='C1', ls='dashed', alpha=1, label='simulated samples (%d)'%len(dcat))
    #if "prot" in outdir:
    #    plt.hist(np.log10(dcat.Prot), density=True, histtype='step', lw=0.6, bins=15, #pbins_prior[::3],
    #color='C1', ls='dashed', alpha=1, label='photometric (%d)'%len(dcat))
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
#fig = corner.corner(mcmc.get_samples()["alphas"][:,30:40], show_titles="%.2f")

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
sigv = 3*0
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

#%%
from scipy.stats import ks_2samp
for sigv in np.linspace(0, 3, 50):
    print (sigv, ks_2samp(_s.ravel(), np.sqrt(dcat.vsini**2-sigv**2)))
