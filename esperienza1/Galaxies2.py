

print("Program start")

from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import emcee
import corner

# Set global font size
plt.rcParams.update({
    'font.size': 16,  # Set the base font size
    'axes.titlesize': 20,  # Title font size
    'axes.labelsize': 20,  # Axis label font size
    'xtick.labelsize': 16,  # X-axis tick label font size
    'ytick.labelsize': 16,  # Y-axis tick label font size
    'legend.fontsize': 18,  # Legend font size
    'figure.titlesize': 22  # Figure title font size
})

print("Loading Table...")
gaia = Table.read('Gaia.vot')
df = gaia.to_pandas()
df = df.loc[(df['b'] >= -5) & (df['b'] <= 5)]
df['l'] = np.deg2rad(df['l'])

print("Computing derived quantities...")
R = 8300
df['cosl'] = np.cos(df['l'])
df['sinl'] = np.sin(df['l'])
df['d'] = 1000 / df['parallax']
df['D'] = np.sqrt(df['d']**2 + R**2 - 2 * df['d'] * R * df['cosl'])
df['sigmad'] = 1000 * df['parallax_error'] / (df['parallax']**2)
df['DER_1/D'] = -df['sinl'] * R * (df['d'] - R * df['cosl']) / (df['D']**3)
df['DER_1/D_wrt_d'] = (-1 / df['D']**3) * (df['d'] - R * df['cosl'])
df['DER_d'] = df['DER_1/D'] * df['DER_1/D_wrt_d']
df['DER2_Prop_model'] = (df['sigmad'] * df['DER_d'])**2
df['VAR_rad_vel'] = df['radial_velocity_error']**2

print("Defining statistical model...")
def v_mod(par, sinl, cosl, D):
    vrot, u, v, VarIntrinsic = par
    return vrot*sinl*((R/D)-1) - u*cosl - v*sinl

def log_like(par, vrad, var_vrad, prop, sinl, cosl, D):
    vrot, u, v, VarIntrinsic = par
    return -0.5 * np.sum(((vrad - v_mod(par, sinl, cosl, D))**2) / (var_vrad + (vrot**2) * prop + VarIntrinsic**2) + np.log(var_vrad + (vrot**2) * prop + VarIntrinsic**2))

def log_prior(par):
    vrot, u, v, VarIntrinsic = par
    if not (-500 <= vrot <= 500 and -500 <= u <= 500 and -500 <= v <= 500 and 0.1 <= VarIntrinsic <= 1e6):
        return -np.inf
    return -(0.5 * (u**2 + v**2) / 200**2) - np.log(VarIntrinsic)

def log_prob(par, vrad, var_vrad, prop, sinl, cosl, D):
    lp = log_prior(par)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(par, vrad, var_vrad, prop, sinl, cosl, D)

print("Initializing MCMC sampling...")
ndim, nwalkers, nsteps, burnin = 4, 96, 10000, 250
p0 = np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(df['radial_velocity'], df['VAR_rad_vel'], df['DER2_Prop_model'], df['sinl'], df['cosl'], df['D']))
print("Running MCMC...")
sampler.run_mcmc(p0, nsteps, progress=True)
flat_samples = sampler.get_chain(discard=burnin, thin=10, flat=True)
print("MCMC finished. Shape of samples:", flat_samples.shape)

print("Plotting results...")
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ["V_rot [Km/s]", "U_sun [Km/s]", "V_sun [Km/s]", "Sigma [Km/s]"]
for i in range(ndim):
    axes[i].plot(sampler.get_chain(discard=burnin)[:, :, i], "k", alpha=0.3)
    axes[i].set_ylabel(labels[i])
axes[-1].set_xlabel("Step number")
plt.show()

fig = corner.corner(flat_samples, labels=labels, truths=np.median(flat_samples, axis=0), truth_color="red")
fig.savefig("PosteriorFull.png", dpi=1000, bbox_inches="tight")
print("Script complete.")
