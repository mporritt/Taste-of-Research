"""
dataInteraction.py by Michael Porritt
Tooks for internal storage of lightcurve data.
"""

import os
import lightkurve
from astropy.io import ascii

import TTVModel, TTVLightCurve


def run(start_epic=None, verbose=True):
    
    for EPIC in get_EPICs(start_epic=start_epic):
        
        campaign_list = []
        for campaign in [5, 16, 18]:
            print("="*100)
            print(f"\t\t\t\tRUNNING EPIC {EPIC} CAMPAIGN {campaign}")
            print("="*100)

            model = read_data(EPIC, campaign)
            if model is not None:
                if model.p_ref < 0.7: 
                    print("Calculated period is less than 0.7 - analysis will not be performed.")
                elif model.num_transits < 2: 
                    print(f"Number of transits is not sufficient to yield meaningful results: {model.num_transits}")
                else:
                    model.optimise(verbose=verbose)
                    campaign_list.append(model)
            del model
            
        if len(campaign_list) > 1:
            TTVModel.TTVModel.stitch_ttvs(campaign_list, save=True)

            
def get_EPICs(start_epic=None):
    with open("Data/EPIC_list.txt", 'r') as f:
        epics = [int(line[:-1]) for line in f.readlines()]

    if start_epic is not None: epics = epics[epics.index(start_epic):]
    
    return epics
        

def read_data(EPIC, campaign, lookup_pars=True, verbose=True):
    """ Retrieve the light curve for the given EPIC and campaign. """
    path = get_path(EPIC, campaign)
    save_path = f"Data/EPIC {EPIC}"
    
    t0_ref, p_ref = None, None
    if lookup_pars:
        t0_ref, p_ref = parameter_lookup(EPIC, verbose=verbose)
    
    if os.path.isfile(path):
        return TTVModel.TTVModel.from_fits(path, p_ref=p_ref, save_path=save_path)
    else:
        if verbose: print(f"File not Found: EPIC {EPIC}, campaign {campaign}.")
    return


def parameter_lookup(search_id, verbose=True):
    search_id = int(search_id)
    table = ascii.read("Data/apjsab346bt7_mrt.txt")

    for id, p, t0 in table.iterrows('ID', 'P', 't0'):
        if int(id) == search_id:
            if verbose: print(f"Found parameters: t0 = {t0 + 167}, p = {p}")
            return (t0+167, p)

    if verbose: print(f"Parameters for system {search_id} not found :(")
    return (None, None)

        
def get_path(EPIC, campaign):
    path = f"Data/EPIC {EPIC}/mastDownload/HLSP/" + \
           f"hlsp_everest_k2_llc_{EPIC}-c{str(campaign).zfill(2)}_kepler_v2.0_lc/" + \
           f"hlsp_everest_k2_llc_{EPIC}-c{str(campaign).zfill(2)}_kepler_v2.0_lc.fits"
    return path


def stitch_all(start_epic=None):
    for EPIC in get_EPICs(start_epic=start_epic):
        stitch_ttvs(EPIC, save=True)
        
    print("All Done Stitching :D")
    

def stitch_ttvs(EPIC, save=False):
    print("="*80)
    print(f"\t\t\tStitching EPIC {EPIC}")
    print("="*80)
    
    models = []
    for campaign in [5, 16, 18]:
        fitsfile = get_path(EPIC, campaign)
        if not os.path.isfile(fitsfile):
            print(f"File not found: {fitsfile}")
            continue
        
        txtfile = f"Data/EPIC {EPIC}/EPIC {EPIC}-c{str(campaign).zfill(2)}_pars.txt"
        if not os.path.isfile(txtfile):
            print(f"File not found: {txtfile}")
            continue
        
        save_path = f"Data/EPIC {EPIC}"
        models.append(TTVModel.TTVModel.from_txt(fitsfile, txtfile, save_path=save_path))
    
    return TTVModel.TTVModel.stitch_ttvs(models, save=save)
    
    
    
        

        
        