"""
TTVModel.py by Michael Porritt
"""

import lightkurve as lk
import exoplanet as xo
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares

import numpy as np
import scipy as sp
import pymc3 as pm
from matplotlib import pyplot as plt

from TTVLightCurve import TTVLightCurve


class TTVModel:
    """
    A model intended to explicitely describe the ttvs (transit timing variations) in a planets transits, as well as its orbital parameters.
    """
    
    def __init__(self, lightcurve, p_ref=None, save_path="."):
        """ 
        Construct a Model given a lightcurve.
        
        Parameters:
            lightcurve -- A lightkurve.LightCurve object, including flux_err values. This must be flattened and it is a good idea to remove 
                          extreme outliers as well.
            p_ref -- Optional. The period of transit. Will be determined using BLS if not supplied.
            t0_ref -- Optional. The time of the first transit. Will be determined using BLS if not supplied.
        """        
        self.lightcurve = lightcurve
        
        # Reference values for period and t0
        t0_ref, p_ref, num_transits = self.get_ref_vals(p_ref=p_ref)
        self.p_ref = p_ref
        self.t0_ref = t0_ref
        self.num_transits = num_transits
        print(f"Reference parameters: t0={t0_ref}, period={p_ref}, num_transits={num_transits}")
        
        self.pars = {}    # Dictionary of orbital model parameters
        self.pars['ttvs'] = [0] * self.num_transits    # List of ttvs for each transit
        self.pars['e_ttvs'] = [0] * self.num_transits
        self.pars['E_ttvs'] = [0] * self.num_transits
        
        self.save_path = save_path
        self.save_name = f"{self.lightcurve.meta['OBJECT']}-c{str(self.lightcurve.meta['CAMPAIGN']).zfill(2)}"
    
    
    @staticmethod
    def from_fits(filename, p_ref=None, save_path=""):
        """ Return a TTVModel given the filename of the everest light curve """
        lc = TTVLightCurve.from_fits(filename)
        lc = lc.flatten()
        lc = lc.remove_outliers(sigma_lower=10, sigma_upper=5)
        
        return TTVModel(lc, p_ref=p_ref, save_path=save_path)
    
    
    @staticmethod
    def stitch_ttvs(TTVModels, save=False):
        """ Stitch together the ttvs of several TTVModel objects. Return a tuple of lists of time vs ttvs + err. """
        
        mean_period = np.mean([model.p_ref for model in TTVModels])
        
        for model in TTVModels:
            model.change_ref_period(mean_period)
        
        time = []
        ttvs = []
        e = []
        E = []
        
        for model in TTVModels:
            for n, ttv in enumerate(model.pars['ttvs']):
                time.append(model.t0_ref + model.p_ref * n + ttv)
                ttvs.append(ttv)
                e.append(model.pars['e_ttvs'][n])
                E.append(model.pars['E_ttvs'][n])
        
        ttvs_mins = [24*60*t for t in ttvs]
        e_mins = [24*60*t for t in e]
        E_mins = [24*60*t for t in E]
    
        plt.figure()
        plt.errorbar(time, ttvs_mins, yerr=(e_mins, E_mins), fmt='.k')
        plt.ylabel("O - C [mins]")
        plt.xlabel("Time [Days]")
        
        if save: 
            save_path = f"{TTVModels[0].save_path}/{TTVModels[0].lightcurve.meta['OBJECT']}_stitched_TTVs.png"
            plt.savefig(save_path)
        plt.show()
        
        return (time, ttvs, e, E)
    
    
    def optimise(self, verbose=True):
        """ Optimise the orbital parameters and TTVs iteratively,  """
        
        plt.figure()
        self.lightcurve.errorbar()
        plt.savefig(f"{self.save_path}/{self.save_name}_lightcurve.png")
        plt.show()
        
        ### TODO: be certain of convergence - use logp values?
        for n in range(3):
            if not verbose: print(f"Loop {n+1} ...")
            self.fit_shape(verbose=verbose)
            self.fit_ttvs(verbose=verbose)
        
        self.fit_shape(run_MCMC=True, verbose=verbose)
        self.fit_ttvs(run_MCMC=True, verbose=verbose)
        
        self.plot_folded(save=True)
        self.plot_ttvs(save=True)
        self.plot_transits_stacked(save=True)
        
        print("Writing graphs and parameters to file.")
        self.write_pars()
        
        
    def fit_shape(self, run_MCMC=False, r_start=None, b_start=None, verbose=True):
        """
        Fit the orbital parameters to the shape of the folded lightcurve data.
        
        Parameters:
            run_MCMC -- Boolean to run Monte Carlo Markov Chain methods. This will take longer but yields better results 
                        and gives uncertainties.
            r_start -- Starting estimate for the relative radius.
            b_start -- Starting estimate for the impact parameter.
        """
        
        if verbose: print("Optimising the shape of the orbital model:")
        
        folded_lc = self.lightcurve.fold(self.p_ref, self.t0_ref, ttvs=self.pars['ttvs'])
        t = folded_lc.time * self.p_ref
        y = folded_lc.flux
        sd = folded_lc.flux_err
        
        if r_start is None: r_start = 0.055
        if b_start is None: b_start = 0.5
        
        with pm.Model() as model:
            mean = pm.Normal("mean", mu=1.0, sd=0.1) # Baseline flux
            t0 = pm.Normal("t0", mu=0, sd=0.025)

            u = xo.distributions.QuadLimbDark("u") # Quadratic limb-darkening parameters
            r = pm.Uniform("r", lower=0.01, upper=0.1, testval=r_start) # radius ratio
            b = xo.distributions.ImpactParameter("b", ror=r, testval=b_start) # Impact parameter

            orbit = xo.orbits.KeplerianOrbit(period=self.p_ref, t0=t0, b=b)

            # Compute the model light curve
            lc = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=r, t=t)
            light_curve = pm.math.sum(lc, axis=-1) + mean

            pm.Deterministic("light_curve", light_curve) # track the value of the model light curve for plotting purposes

            # The likelihood function
            pm.Normal("obs", mu=light_curve, sd=sd, observed=y)
            
            map_soln = xo.optimize(start=model.test_point, verbose=verbose, progress_bar=False)
        
        for k in ['mean', 't0', 'u', 'r', 'b']:
            self.pars[k] = map_soln[k]
            if verbose: print('\t', k, '=', self.pars[k])
        
        if run_MCMC:
            np.random.seed(42)
            with model:
                trace = pm.sample(
                    tune=3000,
                    draws=3000,
                    start=map_soln,
                    cores=1,
                    chains=2,
                    step=xo.get_dense_nuts_step(target_accept=0.9),
                )

            for k in ['mean', 't0', 'u', 'r', 'b']:
                self.pars[k] = np.median(trace[k], axis=0)
                self.pars['e_'+k] =   self.pars[k] - np.percentile(trace[k], 16, axis=0)
                self.pars['E_'+k] = - self.pars[k] + np.percentile(trace[k], 84, axis=0)
                
                if verbose: print(f"\t{k} = {self.pars[k]} /+ {self.pars['E_'+k]} /- {self.pars['e_'+k]}")
            
    
    def fit_ttvs(self, run_MCMC=False, verbose=True):
        """
        Fit transit timing variations to each transit using the shape given by best-fit orbital parameters.
        """
        ttv_start = np.median(self.pars['ttvs']) + self.pars['t0']
        
        for n in range(self.num_transits):
            self.fit_ttv(n, run_MCMC=run_MCMC, ttv_start=ttv_start, verbose=verbose)
        
    
    def fit_ttv(self, n, run_MCMC=False, ttv_start=None, verbose=True):
        """
        Fit a single transit with a transit timing variation, using the shape given by best-fit orbital parameters.
        """
        if verbose: print("Fitting ttv for transit number", n)
        
        # Get the transit lightcurve
        transit = self.lightcurve.get_transit(n, self.p_ref, self.t0_ref)
        t = transit.time * self.p_ref
        y = transit.flux
        sd = transit.flux_err
        
        if ttv_start is None:
            ttv_start = np.median(self.pars['ttvs'])
        
        with pm.Model() as model:
            ttv = pm.Normal("ttv", mu=ttv_start, sd=0.025) # sd = 36 minutes

            orbit = xo.orbits.KeplerianOrbit(
                period=self.p_ref, 
                t0=ttv,
                b=self.pars['b']
            )

            light_curves = xo.LimbDarkLightCurve(self.pars['u']).get_light_curve(
                orbit=orbit, 
                r=self.pars['r'], 
                t=t
            )
            light_curve = pm.math.sum(light_curves, axis=-1) + 1
            pm.Deterministic("transit_"+str(n), light_curve)

            pm.Normal("obs", mu=light_curve, sd=sd, observed=y)

            map_soln = xo.optimize(start=model.test_point, verbose=False, progress_bar=False)
        
        self.pars['ttvs'][n] = float(map_soln['ttv'])
        if verbose: print(f"\t ttv {n} = {self.pars['ttvs'][n]}")
        
        if run_MCMC:
            np.random.seed(42)
            with model:
                trace = pm.sample(
                    tune=500,
                    draws=500,
                    start=map_soln,
                    cores=1,
                    chains=2,
                    step=xo.get_dense_nuts_step(target_accept=0.9),
                )
            
            self.pars['ttvs'][n] = np.median(trace['ttv'])
            self.pars['e_ttvs'][n] =   self.pars['ttvs'][n] - np.percentile(trace['ttv'], 16, axis=0)
            self.pars['E_ttvs'][n] = - self.pars['ttvs'][n] + np.percentile(trace['ttv'], 84, axis=0)
            
            if verbose: print(f"\t ttv {n} = {self.pars['ttvs'][n]} /+ {self.pars['E_ttvs'][n]} /- {self.pars['e_ttvs'][n]}")
    
    
    def get_transit_curve(self, t, t_offset=0):
        
        orbit = xo.orbits.KeplerianOrbit(
            period=self.p_ref, 
            t0=t_offset, 
            b=self.pars['b']
        )
        
        lightcurve = xo.LimbDarkLightCurve(self.pars['u']).get_light_curve(
            orbit=orbit, 
            r=self.pars['r'], 
            t=t
        ).eval() + self.pars['mean']
        
        return lightcurve
    
    
    def plot_ttvs(self, save=False):
        x = np.arange(len(self.pars['ttvs']))
        ttvs = [24*60*t for t in self.pars['ttvs']]
        e = [24*60*t for t in self.pars['e_ttvs']]
        E = [24*60*t for t in self.pars['E_ttvs']]
        
        plt.figure()
        plt.errorbar(x, ttvs, yerr=(e, E), fmt='.k')
        
        plt.xlabel("Transit Number")
        plt.ylabel("O - C [mins]")
        plt.title(f"Transit Timing Variations in {self.lightcurve.label}")
        
        if save: plt.savefig(f"{self.save_path}/{self.save_name}_ttvs.png")
        plt.show()
        
    
    def plot_folded(self, xlim=None, save=False):        
        folded_lc = self.lightcurve.fold(self.p_ref, self.t0_ref, ttvs=self.pars['ttvs'])
        folded_lc.errorbar()
        
        if 't0' in self.pars:
            model = self.get_transit_curve(folded_lc.time*self.p_ref, t_offset=self.pars['t0'])
            plt.plot(folded_lc.time, model, 'c', label="Model light curve")
        
        if xlim is not None:
            plt.xlim(-xlim, xlim)
        
        if save: plt.savefig(f"{self.save_path}/{self.save_name}_folded.png")
        plt.title(f"Folded Flux for {self.lightcurve.label}")
        plt.show()
        
        
    def plot_transit(self, n, xlim=None, save=False):        
        transit = self.lightcurve.get_transit(n, self.p_ref, self.t0_ref)
        transit.errorbar()
        
        if 't0' in self.pars:
            model = self.get_transit_curve(transit.time*self.p_ref, t_offset=self.pars['ttvs'][n])
            plt.plot(transit.time, model, 'c', label="Model light curve")
        
        if xlim is not None:
            plt.xlim(-xlim, xlim)
        
        if save: plt.savefig(f"{self.save_path}/{self.save_name}_transit{n}.png")
        plt.title(f"Transit {n} for {self.lightcurve.label}")
        plt.show()
        
        
    def plot_transits_stacked(self, save=False):
        """ Plot all transits stacked vertically in a single figure. """
        
        fig = plt.figure(figsize=(10, 2.5*self.num_transits)) # , constrained_layout=True
        gs = fig.add_gridspec(self.num_transits, hspace=0)
        ax = gs.subplots(sharex=True, sharey=True)
        
        for n in range(self.num_transits):
            transit = self.lightcurve.get_transit(n, self.p_ref, self.t0_ref)
            ax[n].errorbar(transit.time, transit.flux, yerr=transit.flux_err, 
                           fmt='none', ecolor='k', elinewidth=1)
            
            if 't0' in self.pars:
                model = self.get_transit_curve(transit.time*self.p_ref, t_offset=self.pars['ttvs'][n])
                ax[n].plot(transit.time, model, 'c', label="Model light curve")
            
            ax[n].annotate(f"Transit {n}", (10, 10), xycoords='axes pixels')
            ax[n].set_xlim([-0.5, 0.5])
            ax[n].label_outer()
          
        ax[0].set_title(f"Transits of {self.lightcurve.label}")
        if save: fig.savefig(f"{self.save_path}/{self.save_name}_transits.png")
        plt.show()
    
    
    def change_ref_period(self, p):
        """
        Change the reference value for the period. Update the ttvs accordingly.
        """
        
        for n in range(self.num_transits):
            self.pars['ttvs'][n] = self.pars['ttvs'][n] + n * (self.p_ref - p)

        self.p_ref = p
        return
        
        
    def get_ref_vals(self, p_ref=None):
        t = self.lightcurve.time
        y = self.lightcurve.flux
        dy = self.lightcurve.flux_err

        bls = BoxLeastSquares(t, y, dy)
        durations = [0.05,0.1,0.2]
        if p_ref is None:
            periodogram = bls.autopower(durations)
        else:
            periods = np.linspace(p_ref*0.9, p_ref*1.1, 5000)
            periodogram = bls.power(periods, durations)

        max_power = np.argmax(periodogram.power)
        stats = bls.compute_stats(periodogram.period[max_power],
                                  periodogram.duration[max_power],
                                  periodogram.transit_time[max_power])
        num_transits = len(stats['transit_times'])
        t0 = periodogram.transit_time[max_power]
        p = periodogram.period[max_power]
        
        return (t0, p, num_transits)
        
        
    def write_pars(self):
        """ Write all the parameter values to a file in the save path """
        path = f"{self.save_path}/{self.save_name}_pars.txt"
        
        s = f"""{path} : Parameter values
        p_ref={self.p_ref}
        t0_ref={self.t0_ref}
        """
            
        for k in self.pars:
            s += f"{k}={self.pars[k]}\n"
        
        with open(path, 'w') as f:
            f.write(s)
            
        
        