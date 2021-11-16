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
sns.set(style='ticks', font_scale=1.6, font='times')
#sns.set_palette('colorblind')
plt.rcParams["figure.figsize"] = (18-2,6+1)
from matplotlib import rc
rc('text', usetex=True)

#%%
pmin, pmax = 0.0, 2.0
def read_data(datadir):
    posts = glob.glob(datadir+"gpn100*run5_mcmc.pkl")
    #if not len(posts):
    #    posts = glob.glob(datadir+"gpn100*run5*.pkl")
    filebase = posts[0].replace("/", "_").split(".")[0]
    print (filebase)

    with open(posts[0], 'rb') as f:
        mcmc = dill.load(f)

    priors = np.array(mcmc.get_samples()["priors"])
    q = np.percentile(priors, [16, 50, 84], axis=0)
    q2 = np.percentile(priors, [5, 95], axis=0)
    qmean = np.mean(priors, axis=0)
    qstd = np.std(priors, axis=0)

    logpgrid = mcmc.get_samples()['logpgrid'][0]

    def compute_stats(priors):
        lnpmeans, lnpstds = [], []
        for i in range(len(priors)):
            pi = priors[i]
            _mean = np.average(logpgrid, weights=pi)
            lnpmeans.append(_mean)
            lnpstds.append(np.sqrt(np.average((logpgrid-_mean)**2, weights=pi)))
        lnpmeans = np.array(lnpmeans)
        lnpstds = np.array(lnpstds)
        print ("mean: %.2f +/- %.2f"%(np.mean(lnpmeans), np.std(lnpmeans)))
        print ("std: %.2f +/- %.2f"%(np.mean(lnpstds), np.std(lnpstds)))
        return np.mean(lnpmeans), np.mean(lnpstds)

    mu, std = compute_stats(priors)
    return logpgrid, qmean, qstd, mu, std


#%%
def plot_data(datadirs, labels=None, lines=None, save=None):
    xs, ys, ystds, mus, stds = [], [], [], [], []
    for datadir in datadirs:
        _x, _y, _s, _mu, _std = read_data(datadir)
        xs.append(_x)
        ys.append(_y)
        ystds.append(_s)
        mus.append(_mu)
        stds.append(_std)

    plt.figure()
    plt.xlabel("$\log_{10}P_\mathrm{rot}$ $\mathrm{(days)}$")
    plt.ylabel("$\mathrm{probability\ density}$")
    plt.xlim(pmin, pmax)
    #plt.yscale("log")
    for i,dirn in enumerate(datadirs):
        if labels is None:
            lab = dirn[:-1]
        else:
            lab = labels[i]
        if lines is None:
            ls = 'solid'
        else:
            ls = lines[i]
        #plt.plot(xs[i], ys[i], color="C%d"%i, label=lab.replace("_","\_")+": $%.2f\pm%.2f$"%(mus[i], stds[i]), ls=ls)
        #plt.plot(xs[i], ys[i], color="C%d"%i, label=lab.replace("_","\_")+": mean $%.2f$, SD $%.2f$"%(mus[i], stds[i]), ls=ls)
        plt.plot(xs[i], ys[i], color="C%d"%i, label=lab+": mean $%.2f$, SD $%.2f$"%(mus[i], stds[i]), ls=ls)
        plt.fill_between(xs[i], ys[i]-ystds[i], ys[i]+ystds[i], color="C%d"%i, alpha=0.1)
        #plt.fill_between(logpgrid, q2[0], q2[1], color="C0", alpha=0.05)
    #plt.plot(logpgrid, qmean, color="C0", label=r"recovered (mean \& SD)")
    #plt.fill_between(logpgrid, qmean-qstd, qmean+qstd, color="C0", alpha=0.1)
    #plt.fill_between(logpgrid, q2[0], q2[1], color="C0", alpha=0.05)
    plt.legend(loc="upper left")
    if save is not None:
        plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()


#%%
10**1.07
10**1.16
datadirs = ["all_edr3_upper3_lowmyoung/", "all_edr3_upper3_lowmold/"]
plot_data(datadirs, labels=["$1.05$-$1.15\,M_\odot$, $1$-$4.5\,\mathrm{Gyr}$", "$1.05$-$1.15\,M_\odot$, $4.5$-$7\,\mathrm{Gyr}$"], lines=['solid', 'dashed'], save='all_edr3_upper3_age.png')

#%%
datadirs = ["all_edr3_binflag/", "all_edr3_upper3/", "all_b16macro/", "all_d14macro/"]

#%%
plot_data(datadirs)

#%%
datadirs = ["all_edr3_binflag_t0/", "all_edr3_binflag_t1/", "all_edr3_binflag_t2/"] + ["all_edr3_upper3_t0/", "all_edr3_upper3_t1/", "all_edr3_upper3_t2/"]

#%%
plot_data(datadirs)

#%%
datadirs = ["all_edr3_upper3_t0/", "all_edr3_upper3_t1/", "all_edr3_upper3_t2/"]
plot_data(datadirs, labels=["$5878$-$6103\,\mathrm{K}$", "$6103$-$6246\,\mathrm{K}$", "$6246$-$6643\,\mathrm{K}$"], lines=['solid', 'dashed', 'dotted'], save='all_edr3_upper3_temp.png')

#%%
datadirs = ["all_b16macro_t0/", "all_b16macro_t1/", "all_b16macro_t2/"]

#%%
plot_data(datadirs)

#%%
datadirs = ["all_edr3_binflag_t0/", "all_edr3_binflag_t1/", "all_edr3_binflag_t2/"] + ["all_b16macro_t0/", "all_b16macro_t1/", "all_b16macro_t2/"]

#%%
plot_data(datadirs)
