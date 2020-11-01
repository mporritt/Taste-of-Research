"""
TTVLightCurve.py by Michael Porritt
"""

from astropy.io import fits
import lightkurve
import numpy as np



class TTVLightCurve(lightkurve.LightCurve):
    """
    An extention of lightkurve.LightCurve with additional methods for use in modelling transit timing variations in exoplanets.
    """
    
    @staticmethod
    def from_fits(filename):
        """ Create a new TTVLightCurve from a fits file. """
        
        with fits.open(filename) as hdul:
            data = hdul[1].data
            
            meta = {}
            for k in ['MISSION', 'CAMPAIGN', 'OBJECT', 'OBSMODE']:
                meta[k] = hdul[0].header[k]
        
        label = (meta['OBJECT'] + ', ' + meta['MISSION'] + ' - c' + str(meta['CAMPAIGN']))
        
        return TTVLightCurve(time=data['TIME'], flux=data['FLUX'], flux_err=data['FRAW_ERR'], label=label, meta=meta)
        
    
    def fold(self, period, t0, ttvs=None):
        """
        Folds the lightcurve at a specified period and reference time t0.
        Accounts for transit timing variations if a list of ttvs is given.
        """
        
        x_fold = (self.time - t0 + 0.5 * period) % period - 0.5 * period    # phase not including ttvs
            
        if ttvs is not None:
            for i, t in enumerate(self.time):
                n = int( (t - t0 + 0.5 * period) // period )    # phase number
                if n < 0 or n >= len(ttvs): continue
                        
                x_fold[i] -= ttvs[n]
         
        phase = x_fold / period
        
        inds = np.argsort(phase)
        return lightkurve.FoldedLightCurve(time=phase[inds], flux=self.flux[inds], flux_err=self.flux_err[inds], period=period, t0=t0)
    
    
    def get_transit(self, n, period, t0):
        """
        Create a new LightCurve object that is a window for a single transit, with specified period, t0 and transit number, n.
        """
        
        phase, flux, flux_err = [], [], []
        for i, t in enumerate((self.time - t0 - n*period) / period):
            if (t > -0.5) and (t < 0.5):
                phase.append(t)
                flux.append(self.flux[i])
                flux_err.append(self.flux_err[i])
        
        return lightkurve.FoldedLightCurve(time=phase, flux=flux, flux_err=flux_err)
        
    