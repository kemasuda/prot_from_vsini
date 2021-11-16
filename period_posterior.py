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
#def p_marg_likelihood(p, like_vsini, like_r):
#    integrand = lambda cosi, r: like_vsini(vunit*r/p*np.sqrt(1-cosi**2)) * like_r(r) # y=cosi, x=r
#    return integrate.dblquad(integrand, 0., rmax, lambda x: 0, lambda x: 1)[0]
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

#%%
outdir = "all_edr3_binflag/"
outdir = "all_edr3_upper3/"
outdir = "all_d14macro/"
outdir = "all_b16macro/"
outdir = "sim-lognorm_1_0p2/"
outdir = "sim-loguni_m0p5_1/"
outdir = "sim-2lognorm_1p1_0p1_0p6_0p1/"

#%%
files = glob.glob(outdir+"*_rad.csv")
d = pd.read_csv(files[0])
print (files)
print ("# %d stars without radii."%np.sum(d.rad!=d.rad))

#%%
if "rade" not in d.keys():
    d["rade"] = 0.5*(d.radep + d.radem)

#%%
#if not os.path.exists(outdir):
#    os.system("mkdir %s"%outdir)

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
    # for upper3
    #if vsini < 3:
    #    vsini, sig_vsini = 1, 1.5
    r, sig_r = d.rad, d.rade
    rmin, rmax = max(0, r-5*sig_r), r+5*sig_r
    #like_vsini = lambda x: pdf_norm(x, vsini, sig_vsini)
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

    #try:
    parr, plike = compute_pml(_d)#, parr=np.logspace(-1, 2, 100))
    np.savetxt(outdir+"%s_plike.txt"%str(kic).zfill(9), np.array([parr, plike]).T, fmt='%.8e')

    plt.figure()
    plt.xlabel("$\log_{10}P_\mathrm{rot}$ $\mathrm{(days)}$")
    plt.ylabel("marginal likelihood")
    plt.xlim(logpmin, logpmax)
    plt.plot(np.log10(parr), plike, '-')
    plt.title("KIC %s: radius $%.1f\,R_\odot$, vsini $%.2f\,\mathrm{km/s}$"%(_d.kepid, _d.rad, _d.vsini))
    plt.savefig(outdir+"%s_plike.png"%str(kic).zfill(9), dpi=200, bbox_inches="tight")
    plt.close()
    #except:
    #    pass

#%%
"""
like_vsini = lambda x: pdf_norm(x, 10, 2)
like_r = lambda x: pdf_norm(x, 1, 0.1)

#%%
parr = np.logspace(-1, 2, 100)
like_p_marg = np.array([p_marg_likelihood(_p, like_vsini, like_r) for _p in parr])

#%%
plt.xlabel("$\log_{10}P$ (days)")
plt.ylabel("marginal likelihood")
plt.xlim(-1, 2)
plt.plot(np.log10(parr), like_p_marg, '-')
"""
