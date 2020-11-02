"""
dataInteraction.py by Michael Porritt
Tooks for internal storage of lightcurve data.
"""

import os
import lightkurve
from astropy.io import ascii

import TTVModel, TTVLightCurve


def run(start_epic=None, verbose=True):
    with open("Data/EPIC_list.txt", 'r') as f:
        epics = [int(line[:-1]) for line in f.readlines()]

    if start_epic is not None: epics = epics[epics.index(start_epic):]

    for EPIC in epics:
        campaign_list = []
        for campaign in [5, 16, 18]:
            print("="*100)
            print(f"\t\t\t\tRUNNING EPIC {EPIC} CAMPAIGN {campaign}")
            print("="*100)

            model = read_data(EPIC, campaign)
            if model is not None:
                if model.p_ref > 0.7:
                    model.optimise(verbose=verbose)
                    campaign_list.append(model)
            del model
        TTVModel.TTVModel.stitch_ttvs(campaign_list, save=True)


def read_data(EPIC, campaign, lookup_pars=True):
    """ Retrieve the light curve for the given EPIC and campaign. """
    path = get_path(EPIC, campaign)
    save_path = f"Data/EPIC {EPIC}"
    
    t0_ref, p_ref = None, None
    if lookup_pars:
        t0_ref, p_ref = parameter_lookup(EPIC)
    
    if os.path.isfile(path):
        return TTVModel.TTVModel.from_fits(path, p_ref=p_ref, save_path=save_path)
    else:
        print(f"File not Found: EPIC {EPIC}, campaign {campaign}.")
    return


def parameter_lookup(search_id):
    search_id = int(search_id)
    table = ascii.read("Data/apjsab346bt7_mrt.txt")

    for id, p, t0 in table.iterrows('ID', 'P', 't0'):
        if int(id) == search_id:
            print(f"Found parameters: t0 = {t0 + 167}, p = {p}")
            return (t0+167, p)

    print(f"Parameters for system {search_id} not found :(")
    return (None, None)

        
def get_path(EPIC, campaign):
    path = f"Data/EPIC {EPIC}/mastDownload/HLSP/" + \
           f"hlsp_everest_k2_llc_{EPIC}-c{str(campaign).zfill(2)}_kepler_v2.0_lc/" + \
           f"hlsp_everest_k2_llc_{EPIC}-c{str(campaign).zfill(2)}_kepler_v2.0_lc.fits"
    return path