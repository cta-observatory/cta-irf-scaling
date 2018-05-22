#!/usr/bin/env python
# coding=utf-8

# ===================================================================================
# This script/class script performs the scaling of the CTA calibration data base
# IRF following the given settings. It is intended for the studies of the systematics
# effects in CTA data analysis.
# Author: Ievgen Vovk (2018)
# ===================================================================================

import os
import shutil
import glob
import argparse
import re
import pyfits
import scipy

from matplotlib import pyplot


class CalDB:
    """
    A class to scale the standard CTA IRFs, stored in the CALDB data base in the FITS format.
    """

    def __init__(self, caldb_name, irf_name, verbose=False):
        """
        Constructor of the class. CALDB data bases will be loaded from the library set by "CALDB" environment variable.

        Parameters
        ----------
        caldb_name: string
            CALDB name to use, e.g. '1dc' or 'prod3b'
        irf_name: string
            IRF name to use, e.g. 'North_z20_50h'.
        verbose: bool, optional
            Defines whether to print additional information during the execution.
        """

        self.caldb_path = os.environ['CALDB']
        self.caldb_name = caldb_name
        self.irf = irf_name
        self.verbose = verbose
        
        self.am_ok = True

        self._aeff = dict()
        
        self._check_available_irfs()
        
        self.input_irf_file_name = '{path:s}/data/cta/{caldb:s}/bcf/{irf:s}/irf_file.fits'.format(path=caldb_path,
                                                                                                  caldb=caldb_name,
                                                                                                  irf=irf_name)
        
    def _check_available_irfs(self):
        """
        Internal method that checks which CALDB/IRFs are available in the current library.
        Prints an error if the specified CALDB/IRF combination is not found.

        Returns
        -------
        None
        """

        available_caldb = [path.split('/')[-1] for path in glob.glob(caldb_path + '/data/cta/*')]

        if self.verbose:
            print('-- Available CALDBs -- ')
            print("  {}".format(available_caldb))

        for caldb in available_caldb:
            available_irf = [path.split('/')[-1] for path in glob.glob(caldb_path + '/data/cta/' + caldb + '/bcf/*')]
    
            if self.verbose:
                print("-- Available IRFs for '{:s}' --".format(caldb))
                print("  {}".format(available_irf))
                
        if self.caldb_name not in available_caldb:
            print("ERROR: provided CALDB name '{:s}' is not found!".format(self.caldb_name))
            print('Available options are:')
            print("  {}".format(available_caldb))
            self.am_ok = False
            
        if self.irf not in available_irf:
            print("ERROR: provided IRF name '{:s}' is not found!".format(self.irf))
            print('Available options are:')
            print("  {}".format(available_irf))
            self.am_ok = False
            
    def _scale_psf(self, input_irf_file, psf_scale, n_psf_components=3):
        """
        This internal method scales the IRF PSF extension.

        Parameters
        ----------
        input_irf_file: pyfits.HDUList
            Open pyfits IRF file, which contains the PSF that should be scaled.
        psf_scale: float
            The scale factor. Each PSF sigma (there can be several!) will be multiplied by it.
        n_psf_components: int, optional
            The number of PSF sub-components (gaussians) in the IRF file. Defaults to 3 (typical for prod3b).

        Returns
        -------
        None
        """

        for psf_i in range(0, n_psf_components):
            input_irf_file['POINT SPREAD FUNCTION'].data['SIGMA_{:d}'.format(psf_i+1)] *= psf_scale
            
    def _scale_aeff(self, input_irf_file,
                    aeff_energy_scale, aeff_energy_norm, aeff_energy_transition_width,
                    aeff_theta_scale, aeff_theta_norm, aeff_theta_transition_width):
        """
        This internal method scales the IRF collection area shape. Two scalings can be applied: (1) vs energy and
        (2) vs off-axis angle. In both cases the scaling function is taken as (1 + scale * tanh((x-x0)/dx)). In case
        (1) the scaling value x is log10(energy).

        Parameters
        ----------
        input_irf_file: pyfits.HDUList
            Open pyfits IRF file, which contains the Aeff that should be scaled.
        aeff_energy_scale: float
            Amplitude of the scaling vs energy, must be in [-1;1] range (1 means +/-100% scaling).
            Sign reverses the scaling direction.
        aeff_energy_norm: float
            Energy of the normalization point x0 in TeV.
        aeff_energy_transition_width: float
            Smoothing term dx, defining the sharpness of the transition.
        aeff_theta_scale: float
            Amplitude of the scaling vs off-axis angle, must be in [-1;1] range (1 means +/-100% scaling).
            Sign reverses the scaling direction.
        aeff_theta_norm: float
            Off-axis angle of the normalization point x0 in degrees.
        aeff_theta_transition_width: float
            Smoothing term dx, defining the sharpness of the transition, in degrees.

        Returns
        -------
        None
        """

        # Reading the Aeff parameters
        self._aeff['Elow']     = input_irf_file['Effective area'].data['Energ_lo'][0].copy()
        self._aeff['Ehigh']    = input_irf_file['Effective area'].data['Energ_hi'][0].copy()
        self._aeff['ThetaLow'] = input_irf_file['Effective area'].data['Theta_lo'][0].copy()
        self._aeff['ThetaHi']  = input_irf_file['Effective area'].data['Theta_hi'][0].copy()
        self._aeff['Area']     = input_irf_file['Effective area'].data['EffArea'][0].transpose().copy()
        self._aeff['E']     = scipy.sqrt(self._aeff['Elow']*self._aeff['Ehigh'])
        self._aeff['Theta'] = (self._aeff['ThetaLow']+self._aeff['ThetaHi'])/2.0

        # Creating the energy-theta mesh grid
        energy, theta = scipy.meshgrid(self._aeff['E'], self._aeff['Theta'], indexing='ij')

        # Scaling the Aeff energy dependence
        scaling_val = scipy.log10(energy / aeff_energy_norm) / aeff_energy_transition_width
        self._aeff['Area_new'] = self._aeff['Area'] * (1 + aeff_energy_scale * scipy.tanh(scaling_val))

        # Scaling the Aeff off-axis angle dependence
        scaling_val = (theta - aeff_theta_norm) / aeff_theta_transition_width
        self._aeff['Area_new'] = self._aeff['Area_new'] * (1 + aeff_theta_scale * scipy.tanh(scaling_val))
        
        # Recording the scaled Aeff
        input_irf_file['Effective area'].data['EffArea'][0] = self._aeff['Area_new'].transpose()

    def _append_irf_to_db(self, output_irf_name, output_irf_file_name):
        """
        This internal method appends the new IRF data to the existing calibration data base.

        Parameters
        ----------
        output_irf_name: str
            The name of the IRF to append, e.g. "Aeff_modified". Current IRF name will be added as a prefix.
        output_irf_file_name: str
            Name of the file, which stores the new IRF, e.g. "irf_North_z20_50h_modified.fits"

        Returns
        -------
        None
        """

        db_file_path = '{path:s}/data/cta/{caldb:s}/caldb.indx'.format(path=self.caldb_path, caldb=self.caldb_name)

        # Making a backup
        shutil.copy(db_file_path, db_file_path+'.bak')

        # Opening the database file
        db_file = pyfits.open(db_file_path)

        # Creating a new IRF table which will contain 4 more entries - new PSF/Aeff/Edisp/bkg.
        nrows_orig = len(db_file['CIF'].data)
        nrows_new = nrows_orig + 4
        hdu = pyfits.BinTableHDU.from_columns(db_file['CIF'].columns, nrows=nrows_new)

        # Aeff entry data
        aeff_vals = ['CTA', '1DC', 'NONE', 'NONE', 'ONLINE', 'data/cta/1dc/bcf/' + self.irf, output_irf_file_name,
                     'BCF', 'DATA', 'EFF_AREA', 'NAME({:s})'.format(self.irf + '_' + output_irf_name), 1,
                     '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA effective area']

        # PSF entry data
        psf_vals = ['CTA', '1DC', 'NONE', 'NONE', 'ONLINE', 'data/cta/1dc/bcf/' + self.irf, output_irf_file_name,
                    'BCF', 'DATA', 'RPSF', 'NAME({:s})'.format(self.irf + '_' + output_irf_name), 1,
                    '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA point spread function']

        # Edisp entry data
        edisp_vals = ['CTA', '1DC', 'NONE', 'NONE', 'ONLINE', 'data/cta/1dc/bcf/' + self.irf, output_irf_file_name,
                      'BCF', 'DATA', 'EDISP', 'NAME({:s})'.format(self.irf + '_' + output_irf_name), 1,
                      '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA energy dispersion']

        # Background entry data
        bkg_vals = ['CTA', '1DC', 'NONE', 'NONE', 'ONLINE', 'data/cta/1dc/bcf/'+self.irf, output_irf_file_name,
                    'BCF', 'DATA', 'BKG', 'NAME({:s})'.format(self.irf + '_' + output_irf_name), 1,
                    '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA background']

        # Filling the columns of the new table
        for col_i, colname in enumerate(hdu.columns.names):
            # First fill the previously existing data
            hdu.data[colname][:nrows_orig] = db_file['CIF'].data[colname]
            # Now fill the newly created entries
            hdu.data[colname][nrows_orig+0] = aeff_vals[col_i]
            hdu.data[colname][nrows_orig+1] = psf_vals[col_i]
            hdu.data[colname][nrows_orig+2] = edisp_vals[col_i]
            hdu.data[colname][nrows_orig+3] = bkg_vals[col_i]

        # Replacing the old IRF table
        db_file['CIF'].data = hdu.data

        # Saving the data base
        db_file.writeto(db_file_path, clobber=True)
        db_file.close()

    def scale_irf(self,
                  psf_scale=1.0,
                  aeff_energy_scale=0.0, aeff_energy_norm=1.0, aeff_energy_transition_width=1.0,
                  aeff_theta_scale=0.0, aeff_theta_norm=1.0, aeff_theta_transition_width=1.0,
                  output_irf_file_name=""):
        """
        This method performs scaling of the loaded IRF - both PSF and Aeff, if necessary.
        For the collection area two scalings can be applied: (1) vs energy and
        (2) vs off-axis angle. In both cases the scaling function is taken as
        (1 + scale * tanh((x-x0)/dx)). In case (1) the scaling value x is log10(energy).

        Parameters
        ----------
        psf_scale: float, optional
            The PSF scale factor. Each PSF sigma (there can be several!) will be multiplied by it.
            Defaults to 1.0 - equivalent of no scaling.
        aeff_energy_scale: float, optional
            Amplitude of the scaling vs energy, must be in [-1;1] range (1 means +/-100% scaling).
            Sign reverses the scaling direction.
            Defaults to 0.0 - equivalent of no scaling.
        aeff_energy_norm: float, optional
            Energy of the normalization point x0 in TeV.
            Defaults to 1.0 TeV.
        aeff_energy_transition_width: float, optional
            Smoothing term dx, defining the sharpness of the transition.
            Defaults to 1.0.
        aeff_theta_scale: float
            Amplitude of the scaling vs off-axis angle, must be in [-1;1] range (1 means +/-100% scaling).
            Sign reverses the scaling direction.
            Defaults to 0.0 - equivalent of no scaling.
        aeff_theta_norm: float
            Off-axis angle of the normalization point x0 in degrees.
            Defaults to 1.0 degree.
        aeff_theta_transition_width: float, optional
            Smoothing term dx, defining the sharpness of the transition, in degrees.
            Defaults to 1.0.
        output_irf_file_name: str, optional
            The name of the output IRF file, e.g. 'irf_scaled_version.fits' (the name must follow the "irf_*.fits"
            template). The file will be put to the main directory of the chosen IRF. If empty, the name will be
            automatically generated.
            Defaults to an empty string.

        Returns                         
        -------
        None
        """
        
        if self.am_ok:
            # Opening the IRF input file
            input_irf_file = pyfits.open(self.input_irf_file_name, 'readonly')
            # Scaling the PSF
            self._scale_psf(input_irf_file, psf_scale)
            # Scaling the Aeff
            self._scale_aeff(input_irf_file,
                             aeff_energy_scale, aeff_energy_norm, aeff_energy_transition_width,
                             aeff_theta_scale, aeff_theta_norm, aeff_theta_transition_width)

            # Getting the new IRF and output file names
            if output_irf_file_name == "":
                # No output file name was provided - generating one
                output_psf_part = "P-{:.1f}".format(psf_scale)
                output_aeff_energy_part = "A-{:.1f}-{:.1f}-{:.1f}".format(aeff_energy_scale,
                                                                          aeff_energy_norm,
                                                                          aeff_energy_transition_width)
                output_aeff_theta_part = "{:.1f}-{:.1f}-{:.1f}".format(aeff_theta_scale,
                                                                       aeff_theta_norm,
                                                                       aeff_theta_transition_width)
                # IRF name
                output_irf_name = output_psf_part + "_" + output_aeff_energy_part + output_aeff_theta_part
                # Output file name
                output_irf_file_name = "irf_{:s}.fits".format(output_irf_name)
            else:
                # Output file name was provided. Will chunk the IRF name out of it.
                output_irf_name = re.findall("irf_(.+).fits", output_irf_file_name)[0]

            # Figuring out the output path
            output_path = '{path:s}/data/cta/{caldb:s}/bcf/{irf:s}'.format(path=self.caldb_path,
                                                                           caldb=self.caldb_name,
                                                                           irf=self.irf)

            # Writing the scaled IRF
            input_irf_file.writeto(output_path + "/" + output_irf_file_name, clobber=True)

            # Updating the calibration data base with the new IRF
            self._append_irf_to_db(output_irf_name, output_irf_file_name)
        else:
            print("ERROR: something's wrong with the CALDB/IRF names. So can not update the data base.")

    def get_aeff_scale_map(self):
        """
        This method returns the Aeff scale map, which can be useful for check of the used settings.
        Must be run after the scale_irf() method.

        Returns
        -------
        dict:
            A dictionary with the Aeff scale map.
        """

        scale_map = dict()
        
        scale_map['E_edges'] = scipy.concatenate((self._aeff['Elow'], [self._aeff['Ehigh'][-1]]))
        scale_map['Theta_edges'] = scipy.concatenate((self._aeff['ThetaLow'], [self._aeff['ThetaHi'][-1]]))

        scale_map['Map'] = self._aeff['Area_new'] / self._aeff['Area']
        wh_nan = scipy.where(scipy.isnan(scale_map['Map']))
        scale_map['Map'][wh_nan] = 0
        scale_map['Map'] -= 1
        
        return scale_map

    def plot_aeff_scale_map(self, vmin=-0.5, vmax=0.5):
        """
        This method plots the Aeff scale map, which can be useful for check of the used settings.
        Must be run after the scale_irf() method.

        Parameters
        ----------
        vmin: float, optional
            Minimal scale to plot. Defaults to -0.5.
        vmax: float, optional
            Maximal scale to plot. Defaults to 0.5.

        Returns
        -------
        None
        """

        scale_map = self.get_aeff_scale_map()

        pyplot.semilogx()

        pyplot.xlabel('Energy, TeV')
        pyplot.ylabel('Off-center angle, deg')
        pyplot.pcolormesh(scale_map['E_edges'], scale_map['Theta_edges'], scale_map['Map'].transpose(),
                          cmap='bwr', vmin=vmin, vmax=vmax)
        pyplot.colorbar()
        

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
    arg_parser.add_argument("--psf_scale", default=1.0,
                            help='The PSF scale factor. Each PSF sigma (there can be several!) will be multiplied by it.')
    arg_parser.add_argument("--aeff_energy_scale", default=0.0,
                            help='Amplitude of the scaling vs energy, must be in [-1;1] range (1 means +/-100%% scaling).')
    arg_parser.add_argument("--aeff_energy_norm", default=1.0,
                            help='Energy of the normalization point in TeV.')
    arg_parser.add_argument("--aeff_energy_transition_width", default=1.0,
                            help='Smoothing term, defining the sharpness of the transition.')
    arg_parser.add_argument("--aeff_theta_scale", default=0.0,
                            help='Amplitude of the scaling vs off-axis angle, must be in [-1;1] range (1 means +/-100%% scaling).')
    arg_parser.add_argument("--aeff_theta_norm", default=1.0,
                            help='Off-axis angle of the normalization point in degrees.')
    arg_parser.add_argument("--aeff_theta_transition_width", default=1.0,
                            help='Smoothing term, defining the sharpness of the transition, in degrees.')
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
    # --------------------------------------

    if cmd_args.caldb == '' or cmd_args.irf == '':
        print("CALDB or IRF names were not specified. Exiting... ")
        print("Please read the help!")
        exit()
    else:
        caldb = cmd_args.caldb
        irf = cmd_args.irf
        psf_scale = cmd_args.psf_scale
        aeff_energy_scale = float(cmd_args.aeff_energy_scale)
        aeff_energy_norm = float(cmd_args.aeff_energy_norm)
        aeff_energy_transition_width = float(cmd_args.aeff_energy_transition_width)
        aeff_theta_scale = float(cmd_args.aeff_theta_scale)
        aeff_theta_norm = float(cmd_args.aeff_theta_norm)
        aeff_theta_transition_width = float(cmd_args.aeff_theta_transition_width)
        output_irf_file_name = cmd_args.output_irf_file_name
        plot_aeff_scale_map = cmd_args.plot_aeff_scale_map

        print("")
        print("=== Was called with the following settings ===")
        print("  CALDB name:                   {:s}".format(caldb))
        print("  IRF name:                     {:s}".format(irf))
        print("  PSF scale:                    {:.2f}".format(psf_scale))
        print("  Aeff energy scale:            {:.1f}".format(aeff_energy_scale))
        print("  Aeff energy norm:             {:.1f}".format(aeff_energy_norm))
        print("  Aeff energy transition_width: {:.1f}".format(aeff_energy_transition_width))
        print("  Aeff theta scale:             {:.1f}".format(aeff_theta_scale))
        print("  Aeff theta norm:              {:.1f}".format(aeff_theta_norm))
        print("  Aeff theta transition_width:  {:.1f}".format(aeff_theta_transition_width))
        print("  Output IRF file name:         '{:s}'".format(output_irf_file_name))
        print("  Plot Aeff scale map:          {}".format(plot_aeff_scale_map))
        print("")
        
        caldb_path = os.environ['CALDB']
        
        caldb = CalDB(caldb, irf, verbose=False)
        caldb.scale_irf(psf_scale,
                        aeff_energy_scale, aeff_energy_norm, aeff_energy_transition_width,
                        aeff_theta_scale, aeff_theta_norm, aeff_theta_transition_width, output_irf_file_name)
        
        if plot_aeff_scale_map:
            pyplot.clf()
            caldb.plot_aeff_scale_map()
            pyplot.show()
