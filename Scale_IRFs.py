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
    arg_parser.add_argument("--caldb", default='prod3b',
                            help='Calibration data base name, e.g. "1dc"')
    arg_parser.add_argument("--irf", default='',
                            help='The IRF to scale, e.g. "North_z20_50h"')
    arg_parser.add_argument("--obs2scale", default="",
                            help="Observable involved in the scaling: choose between 'energy' or 'arrival_dir'.")
    arg_parser.add_argument("--err_func_type", default="",
                            help="Error function type: choose among 'constant', 'gradient', 'step'.")
    arg_parser.add_argument("--const_scale", default=1.0,
                            help="Constant error function value. Default = 1.0.")
    arg_parser.add_argument("--psf_scale", default=1.0,
                            help='The PSF scale factor. Each PSF sigma (there can be several!) will be multiplied by it.')
    arg_parser.add_argument("--output_irf_file_name", default="",
                            help="""The name of the output IRF file, e.g. 'irf_scaled_version.fits' (the name must follow
                           the "irf_*.fits" template). The file will be put to the main directory of the chosen IRF. 
                           If empty, the name will be automatically generated.
                           """)
    arg_parser.add_argument("--plot_aeff_scale_map", default=True,
                            help="Defines whether the resulting Aeff scale map should be displayed.")
    arg_parser.add_argument("--verbose", default=False,
                            help="Defines whether to print additional information during the run.")
    cmd_args = arg_parser.parse_args()

    # --------   Variables (to be loaded from a config file)  --------------
    e_transition1 = 0.15  # Tev
    e_transition2 = 5.0  # Tev
    e_min = 0.02  # TeV
    e_max_north = 0.05  # TeV
    e_max_south = 0.3  # TeV
    e_res1 = 0.11  # to be determined from IRF
    e_res2 = 0.06  # to be determined from IRF
    theta_max_north = 7.6  # deg
    theta_max_south = 9.3  # deg
    theta_transition1 = 4.3  # deg
    theta_transition2 = 7.6  # deg
    sigma_theta1 = 0.06  # deg # to be determined from IRF
    sigma_theta2 = 0.06  # deg # to be determined from IRF
    epsilon_aeff = 0.5

    # -----------------------------------------------
    if cmd_args.caldb == '' or cmd_args.irf == '':
        print("CALDB or IRF names were not specified. Exiting... ")
        print("Please read the help!")
        exit()
    else:
        caldb = cmd_args.caldb
        irf = cmd_args.irf
        obs2scale = cmd_args.obs2scale
        err_func_type = cmd_args.err_func_type
        const_scale = float(cmd_args.const_scale)
        psf_scale = float(cmd_args.psf_scale)
        output_irf_file_name = cmd_args.output_irf_file_name
        plot_aeff_scale_map = cmd_args.plot_aeff_scale_map

        # Extract the hemisphere from the IRF name:
        if irf.find("North") != -1:
            hemisphere = "North"
        else:
            hemisphere = "South"
        print(hemisphere)

        # Checks:
        if irf.find("North") == -1 and irf.find("South") == -1:
            print("Hemisphere not specified in IRF name. Exiting...")
            exit()

        if obs2scale == '' or (obs2scale != 'energy' and obs2scale != 'arrival_dir'):
            print("Observable for scaling was not properly specified. Exiting... ")
            print("Please read the help!")
            exit()

        if err_func_type == '' or (
                err_func_type != 'constant' and err_func_type != 'gradient' and err_func_type != 'step'):
            print("Error function type was not properly specified. Exiting... ")
            print("Please read the help!")
            exit()

        if obs2scale == 'arrival_dir' and err_func_type == 'constant':
            print(
                "Constant error function is not implemented for the arrival direction. Choose 'gradient' or 'step'. Exiting... ")
            exit()

        print("")
        print("=== Was called with the following settings ===")
        print("  CALDB name:            {:s}".format(caldb))
        print("  IRF name:              {:s}".format(irf))
        print("  Observable to scale:   {:s}".format(obs2scale))
        print("  ERR function type:     {:s}".format(err_func_type))
        print("  PSF scale:             {:.2f}".format(psf_scale))
        print("  Epsilon:               {:.1f}".format(epsilon_aeff))
        print("")

        # Scale IRF
        caldb_path = os.environ['CALDB']

        caldb = CalDB(caldb, irf, verbose=False)

        # Scale +epsilon
        caldb.scale_irf(e_transition1, e_transition2, e_min, e_max_north, e_max_south,
                        theta_transition1, theta_transition2, sigma_theta1, sigma_theta2,
                        epsilon_aeff,
                        hemisphere, obs2scale, err_func_type, const_scale,
                        psf_scale,
                        output_irf_file_name)

        # Scale -epsilon
        caldb.scale_irf(e_transition1, e_transition2, e_min, e_max_north, e_max_south,
                        theta_transition1, theta_transition2, sigma_theta1, sigma_theta2,
                        -epsilon_aeff,
                        hemisphere, obs2scale, err_func_type, const_scale,
                        psf_scale,
                        output_irf_file_name)

        # Plot
        if plot_aeff_scale_map:
            pyplot.clf()
            caldb.plot_aeff_scale_map()
            pyplot.show()
