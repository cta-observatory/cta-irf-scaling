#!/usr/bin/env python
# coding=utf-8

# ===================================================================================
# This script/class script performs the scaling of the CTA calibration data base
# IRF following the given settings. It is intended for the studies of the systematics
# effects in CTA data analysis.
# Author: Ievgen Vovk (2018)
# ===================================================================================

import os
import argparse
import yaml

from matplotlib import pyplot

from caldb_scaler import CalDB

# =================
# === Main code ===
# =================

if __name__ == "__main__":
    # --------------------------------------
    # *** Parsing command line arguments ***
    arg_parser = argparse.ArgumentParser(description="""
                                                    This script performs the scaling of the CTA calibration data base
                                                    IRF following the given settings. It is intended for the studies
                                                    of the systematics effects in CTA data analysis.
                                                    Please note, that the existing data base will be updated with the 
                                                    newly scaled IRF, i.e. your 'caldb.indx' file will be over-written. 
                                                    Though a back up copy is created each time the script is run, 
                                                    consider creating a master back up manually just in case.
                                                    """)
    arg_parser.add_argument("--config", default='config.yaml',
                            help='Path to the configuration file.')
    arg_parser.add_argument("--plot_aeff_scale_map", default=True,
                            help="Defines whether the resulting Aeff scale map should be displayed.")
    arg_parser.add_argument("--verbose", default=False,
                            help="Defines whether to print additional information during the run.")
    cmd_args = arg_parser.parse_args()
    # -----------------------------------------------

    plot_aeff_scale_map = cmd_args.plot_aeff_scale_map

    # Scaling settings
    config = yaml.load(open(cmd_args.config, 'r'))

    # -------------------------
    print("")
    print("=== Was called with the following settings ===")
    print("  General:")
    for key in config['general']:
        print("    {:.<25s}:  {:s}".format(key, config['general'][key]))
    print("")
    # -------------------------

    # -------------------------
    print("  Collection area:")
    print("    Energy scaling:")

    err_func_type = config['aeff']['energy_scaling']['err_func_type']
    scale_params = config['aeff']['energy_scaling'][err_func_type]

    print("      {:.<23s}:  {:s}".format('err_func_type', err_func_type))
    for key in scale_params:
        print("      {:.<23s}:  {}".format(key, scale_params[key]))

    print("    Angular scaling:")

    err_func_type = config['aeff']['angular_scaling']['err_func_type']
    scale_params = config['aeff']['angular_scaling'][err_func_type]

    print("      {:.<23s}:  {:s}".format('err_func_type', err_func_type))
    for key in scale_params:
        print("      {:.<23s}:  {}".format(key, scale_params[key]))

    print("")
    # -------------------------
    
    # -------------------------
    print("  Background:")
    print("    Energy scaling:")

    err_func_type = config['bkg']['energy_scaling']['err_func_type']
    scale_params = config['bkg']['energy_scaling'][err_func_type]

    print("      {:.<23s}:  {:s}".format('err_func_type', err_func_type))
    for key in scale_params:
        print("      {:.<23s}:  {}".format(key, scale_params[key]))

    print("    Angular scaling:")

    err_func_type = config['bkg']['angular_scaling']['err_func_type']
    scale_params = config['bkg']['angular_scaling'][err_func_type]

    print("      {:.<23s}:  {:s}".format('err_func_type', err_func_type))
    for key in scale_params:
        print("      {:.<23s}:  {}".format(key, scale_params[key]))

    print("")
    # -------------------------
    
    # -------------------------
    print("  PSF:")
    print("    Energy scaling:")

    err_func_type = config['psf']['energy_scaling']['err_func_type']
    scale_params = config['psf']['energy_scaling'][err_func_type]

    print("      {:.<23s}:  {:s}".format('err_func_type', err_func_type))
    for key in scale_params:
        print("      {:.<23s}:  {}".format(key, scale_params[key]))

    print("    Angular scaling:")

    err_func_type = config['psf']['angular_scaling']['err_func_type']
    scale_params = config['psf']['angular_scaling'][err_func_type]

    print("      {:.<23s}:  {:s}".format('err_func_type', err_func_type))
    for key in scale_params:
        print("      {:.<23s}:  {}".format(key, scale_params[key]))
        
    print("")
    # -------------------------

    # -------------------------
    print("  Energy Dispersion:")
    print("    Energy scaling:")

    err_func_type = config['edisp']['energy_scaling']['err_func_type']
    scale_params = config['edisp']['energy_scaling'][err_func_type]

    print("      {:.<23s}:  {:s}".format('err_func_type', err_func_type))
    for key in scale_params:
        print("      {:.<23s}:  {}".format(key, scale_params[key]))

    #print("    Angular scaling:")

    #err_func_type = config['edisp']['angular_scaling']['err_func_type']
    #scale_params = config['edisp']['angular_scaling'][err_func_type]

    #print("      {:.<23s}:  {:s}".format('err_func_type', err_func_type))
    #for key in scale_params:
    #    print("      {:.<23s}:  {}".format(key, scale_params[key]))
        
    print("")
    # -------------------------

    # Scale IRF
    caldb_path = os.environ['CALDB']

    caldb = CalDB(config['general']['caldb'], config['general']['irf'], verbose=False)

    caldb.scale_irf(config)

    # Plot
    if plot_aeff_scale_map:
        pyplot.figure(figsize=(14, 8))
        pyplot.clf()
        
        pyplot.subplot(231)
        caldb.plot_aeff_scale_map()
        
        pyplot.subplot(232)
        caldb.plot_psf_scale_map()
        
        pyplot.subplot(233)
        caldb.plot_edisp_scale_map()
        
        pyplot.subplot(234)
        caldb.plot_bkg_scale_map_detx()
        
        pyplot.subplot(235)
        caldb.plot_bkg_scale_map_dety()
        
        pyplot.tight_layout()
        pyplot.show()
