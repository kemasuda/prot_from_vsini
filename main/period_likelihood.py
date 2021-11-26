#%% Packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import R_jup, R_sun, R_earth, au, M_sun
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
import os, glob

#%%
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='sans-serif')
#sns.set_palette('colorblind')
plt.rcParams["figure.figsize"] = (18,6)

#%%
vunit = (2*np.pi*R_sun/86400).value*1e-3
def integrand_cosi(r, p, like_vsini):
    return lambda x: like_vsini(vunit*r/p*np.sqrt(1-x**2))

def p_marg_likelihood(p, like_vsini, like_r, rmin=0, rmax=3.):
    integrand_r = lambda r: like_r(r) * quad(integrand_cosi(r, p, like_vsini), 0, 1)[0]
    return quad(integrand_r, rmin, rmax)[0]

#%%
logpmin, logpmax = -0.5, 2.5
Ngrid = 1000
pgrid = np.logspace(logpmin, logpmax, Ngrid)

#%% treat vsini<3 as upper limits
upper3 = True*0

#%%
outdir = "period_likelihood/"
if upper3:
    outdir = outdir[:-1] + "_upper3/"

#%%
files = glob.glob("data/sample_stars.csv")
d = pd.read_csv(files[0])
print (files)
print ("# %d stars without radii."%np.sum(d.rad!=d.rad))

#%%
if "rade" not in d.keys():
    d["rade"] = 0.5*(d.radep + d.radem)

#%%
if not os.path.exists(outdir):
    os.system("mkdir %s"%outdir)

#%%
def pdf_norm(x, mu, sigma):
    return np.exp(-0.5*(x-mu)**2/sigma**2)/np.sqrt(2*np.pi)/sigma

def get_like_vsini(vsini, sig_vsini):
    if vsini <= sig_vsini:
        return lambda x: np.heaviside(2*sig_vsini-x, 0)*(0.5/sig_vsini)
        #return lambda x: pdf_norm(x, 0, 2*sig_vsini)*2
    else:
        c = norm.sf(0, vsini, sig_vsini)
        return lambda x: pdf_norm(x, vsini, sig_vsini)/c

def compute_pml(d, parr=pgrid):
    vsini, sig_vsini = d.vsini, 1.
    # treat vsini<3 as upper limits
    if upper3 and vsini < 3:
        vsini, sig_vsini = 1, 1.5
    r, sig_r = d.rad, d.rade
    rmin, rmax = max(0, r-5*sig_r), r+5*sig_r
    like_vsini = get_like_vsini(vsini, sig_vsini)
    like_r = lambda x: pdf_norm(x, r, sig_r)
    like_p_marg = np.array([p_marg_likelihood(_p, like_vsini, like_r, rmin=rmin, rmax=rmax) for _p in parr])
    return parr, like_p_marg

#%%
print (outdir)
for i in range(len(d)):
    _d = d.iloc[i]
    kic = int(_d.kepid)
    print (kic)

    parr, plike = compute_pml(_d)
    np.savetxt(outdir+"%s_plike.txt"%str(kic).zfill(9), np.array([parr, plike]).T, fmt='%.8e')

    plt.figure()
    plt.xlabel("$\log_{10}P_\mathrm{rot}$ $\mathrm{(days)}$")
    plt.ylabel("relative marginal likelihood")
    plt.xlim(logpmin, logpmax)
    plt.plot(np.log10(parr), plike, '-')
    plt.title("KIC %s: radius $%.1f\,R_\odot$, vsini $%.2f\,\mathrm{km/s}$"%(_d.kepid, _d.rad, _d.vsini))
    plt.savefig(outdir+"%s_plike.png"%str(kic).zfill(9), dpi=200, bbox_inches="tight")
    plt.close()
