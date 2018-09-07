import os
import shutil
import glob
import re
import scipy
import astropy.io.fits as pyfits
from matplotlib import pyplot

from scaling_functions import *

# ========================
# ===   Class CalDB   ===
# ========================


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
        self._psf = dict()

        self._check_available_irfs()

        self.input_irf_file_name = '{path:s}/data/cta/{caldb:s}/bcf/{irf:s}/irf_file.fits'.format(path=self.caldb_path,
                                                                                                  caldb=self.caldb_name,
                                                                                                  irf=irf_name)

    def _check_available_irfs(self):
        """
        Internal method that checks which CALDB/IRFs are available in the current library.
        Prints an error if the specified CALDB/IRF combination is not found.

        Returns
        -------
        None
        """

        available_caldb = [path.split('/')[-1] for path in glob.glob(self.caldb_path + '/data/cta/*')]

        if self.verbose:
            print('-- Available CALDBs -- ')
            print("  {}".format(available_caldb))

        for caldb in available_caldb:
            available_irf = [path.split('/')[-1] for path in glob.glob(self.caldb_path + '/data/cta/' + caldb + '/bcf/*')]

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

    def _scale_psf(self, input_irf_file, config):
        """
        This internal method scales the IRF PSF extension.

        Parameters
        ----------
        input_irf_file: pyfits.HDUList
            Open pyfits IRF file, which contains the PSF that should be scaled.
        config: dict
            A dictionary with the scaling settings. Must have following keys defined:
            "energy_scaling": dict
                Contains setting for the energy scaling (see the structure below).
            "angular_scaling": dict
                Contains setting for the off-center angle scaling (see the structure below).

            In both cases, internally the above dictionaries should contain:
            "err_func_type": str
                The name of the scaling function to use. Accepted values are: "constant",
                "gradient" and "step".

            If err_func_type == "constant":
                scale: float
                    The scale factor. passing 1.0 results in no scaling.

            If err_func_type == "gradient":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                range_min: float
                    The x value (energy or off-center angle), that corresponds to -1 scale.
                range_max: float
                    The x value (energy or off-center angle), that corresponds to +1 scale.

            If err_func_type == "step":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                transition_pos: list
                    The list of x values (energy or off-center angle), at which
                    step-like transitions occur. If scaling the energy dependence,
                    values must be in TeVs, if angular - in degrees.
                transition_widths: list
                    The list of step-like transition widths, that correspond to transition_pos.
                    For energy scaling the widths must be in log10 scale.

        Returns
        -------
        None

        """

        # Find all "sigma" values - tells how many PSF components we have in the IRF file
        column_names = [col.name.lower() for col in input_irf_file['POINT SPREAD FUNCTION'].columns]
        sigma_columns = list(filter(lambda s: "sigma" in s.lower(), column_names))

        # --------------------------
        # Reading the PSF parameters
        self._psf = dict()
        self._psf['Elow'] = input_irf_file['POINT SPREAD FUNCTION'].data['Energ_lo'][0].copy()
        self._psf['Ehigh'] = input_irf_file['POINT SPREAD FUNCTION'].data['Energ_hi'][0].copy()
        self._psf['ThetaLow'] = input_irf_file['POINT SPREAD FUNCTION'].data['Theta_lo'][0].copy()
        self._psf['ThetaHi'] = input_irf_file['POINT SPREAD FUNCTION'].data['Theta_hi'][0].copy()

        for i in range(0, len(sigma_columns)):
            sigma_name = 'sigma_{:d}'.format(i + 1)
            self._psf[sigma_name] = input_irf_file['POINT SPREAD FUNCTION'].data[sigma_name][0].transpose().copy()

        self._psf['E'] = scipy.sqrt(self._psf['Elow'] * self._psf['Ehigh'])
        self._psf['Theta'] = (self._psf['ThetaLow'] + self._psf['ThetaHi']) / 2.0
        # --------------------------

        # Creating the energy-theta mesh grid
        energy, theta = scipy.meshgrid(self._psf['E'], self._psf['Theta'], indexing='ij')

        # ---------------------------------
        # Scaling the PSF energy dependence

        # Constant error function
        if config['energy_scaling']['err_func_type'] == "constant":
            scale_params = config['energy_scaling']["constant"]
            # Constant scaling. Loop over all "sigma" values and scale them by the same factor.
            for sigma_column in sigma_columns:
                self._psf[sigma_column + '_new'] = scale_params['scale'] * self._psf[sigma_column]

        # Gradients error function
        elif config['energy_scaling']['err_func_type'] == "gradient":
            scale_params = config['energy_scaling']["gradient"]
            for sigma_column in sigma_columns:
                self._psf[sigma_column + '_new'] = self._psf[sigma_column] * (
                        1 + scale_params['scale'] * gradient(scipy.log10(energy),
                                                             scipy.log10(scale_params['range_min']),
                                                             scipy.log10(scale_params['range_max']))
                )

        # Step error function
        elif config['energy_scaling']['err_func_type'] == "step":
            scale_params = config['energy_scaling']["step"]
            break_points = list(zip(scipy.log10(scale_params['transition_pos']),
                                    scale_params['transition_widths']))

            for sigma_column in sigma_columns:
                self._psf[sigma_column + '_new'] = self._psf[sigma_column] * (
                        1 + scale_params['scale'] * step(scipy.log10(energy), break_points)
                )

        else:
            raise ValueError("Unknown PSF scaling function {:s}"
                             .format(config['energy_scaling']['err_func_type']))
        # ---------------------------------

        # ---------------------------------
        # Scaling the PSF angular dependence

        # Constant error function
        if config['angular_scaling']['err_func_type'] == "constant":
            scale_params = config['angular_scaling']["constant"]
            # Constant scaling. Loop over all "sigma" values and scale them by the same factor.
            for sigma_column in sigma_columns:
                # input_irf_file['POINT SPREAD FUNCTION'].data[sigma_column] *= scale_params['scale']
                self._psf[sigma_column + '_new'] = scale_params['scale'] * self._psf[sigma_column + '_new']

        # Gradients error function
        elif config['angular_scaling']['err_func_type'] == "gradient":
            scale_params = config['angular_scaling']["gradient"]
            for sigma_column in sigma_columns:
                self._psf[sigma_column + '_new'] = self._psf[sigma_column + '_new'] * (
                        1 + scale_params['scale'] * gradient(theta,
                                                             scale_params['range_min'],
                                                             scale_params['range_max'])
                )

        # Step error function
        elif config['angular_scaling']['err_func_type'] == "step":
            scale_params = config['angular_scaling']["step"]
            break_points = list(zip(scale_params['transition_pos'],
                                    scale_params['transition_widths']))

            for sigma_column in sigma_columns:
                self._psf[sigma_column + '_new'] = self._psf[sigma_column + '_new'] * (
                        1 + scale_params['scale'] * step(theta, break_points)
                )

        else:
            raise ValueError("Unknown PSF scaling function {:s}"
                             .format(config['angular_scaling']['err_func_type']))
        # ---------------------------------

        # Recording the scaled PSF
        for i in range(0, len(sigma_columns)):
            sigma_name = 'sigma_{:d}'.format(i + 1)

            input_irf_file['POINT SPREAD FUNCTION'].data[sigma_name][0] = self._psf[sigma_name + '_new'].transpose()

    def _scale_aeff(self, input_irf_file, config):
        """
        This internal method scales the IRF collection area shape.
        Two scalings can be applied: (1) vs energy and (2) vs off-axis angle. In both cases
        the scaling function is taken as (1 + scale * tanh((x-x0)/dx)). In case (1) the scaling
        is performed in log-energy.

        Parameters
        ----------
        input_irf_file: pyfits.HDUList
            Open pyfits IRF file, which contains the Aeff that should be scaled.

        config: dict
            A dictionary with the scaling settings. Must have following keys defined:
            "energy_scaling": dict
                Contains setting for the energy scaling (see the structure below).
            "angular_scaling": dict
                Contains setting for the off-center angle scaling (see the structure below).

            In both cases, internally the above dictionaries should contain:
            "err_func_type": str
                The name of the scaling function to use. Accepted values are: "constant",
                "gradient" and "step".

            If err_func_type == "constant":
                scale: float
                    The scale factor. passing 1.0 results in no scaling.

            If err_func_type == "gradient":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                range_min: float
                    The x value (energy or off-center angle), that corresponds to -1 scale.
                range_max: float
                    The x value (energy or off-center angle), that corresponds to +1 scale.

            If err_func_type == "step":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                transition_pos: list
                    The list of x values (energy or off-center angle), at which
                    step-like transitions occur. If scaling the energy dependence,
                    values must be in TeVs, if angular - in degrees.
                transition_widths: list
                    The list of step-like transition widths, that correspond to transition_pos.
                    For energy scaling the widths must be in log10 scale.

        Returns
        -------
        None

        """

        # Reading the Aeff parameters
        self._aeff['Elow'] = input_irf_file['Effective area'].data['Energ_lo'][0].copy()
        self._aeff['Ehigh'] = input_irf_file['Effective area'].data['Energ_hi'][0].copy()
        self._aeff['ThetaLow'] = input_irf_file['Effective area'].data['Theta_lo'][0].copy()
        self._aeff['ThetaHi'] = input_irf_file['Effective area'].data['Theta_hi'][0].copy()
        self._aeff['Area'] = input_irf_file['Effective area'].data['EffArea'][0].transpose().copy()
        self._aeff['E'] = scipy.sqrt(self._aeff['Elow'] * self._aeff['Ehigh'])
        self._aeff['Theta'] = (self._aeff['ThetaLow'] + self._aeff['ThetaHi']) / 2.0

        # Creating the energy-theta mesh grid
        energy, theta = scipy.meshgrid(self._aeff['E'], self._aeff['Theta'], indexing='ij')

        # ----------------------------------
        # Scaling the Aeff energy dependence

        # Constant error function
        if config['energy_scaling']['err_func_type'] == "constant":
            self._aeff['Area_new'] = self._aeff['Area'] * config['energy_scaling']['constant']['scale']

        # Gradients error function
        elif config['energy_scaling']['err_func_type'] == "gradient":
            scaling_params = config['energy_scaling']['gradient']
            self._aeff['Area_new'] = self._aeff['Area'] * (
                    1 + scaling_params['scale'] * gradient(scipy.log10(energy),
                                                           scipy.log10(scaling_params['range_min']),
                                                           scipy.log10(scaling_params['range_max']))
            )
            
        # Step error function
        elif config['energy_scaling']['err_func_type'] == "step":
            scaling_params = config['energy_scaling']['step']
            break_points = list(zip(scipy.log10(scaling_params['transition_pos']),
                                    scaling_params['transition_widths']))
            self._aeff['Area_new'] = self._aeff['Area'] * (
                    1 + scaling_params['scale'] * step(scipy.log10(energy), break_points)
            )
        else:
            raise ValueError("Aeff energy scaling: unknown scaling function type '{:s}'"
                             .format(config['energy_scaling']['err_func_type']))
        # ----------------------------------

        # ------------------------------------------
        # Scaling the Aeff off-axis angle dependence

        # Constant error function
        if config['angular_scaling']['err_func_type'] == "constant":
            self._aeff['Area_new'] = self._aeff['Area_new'] * config['angular_scaling']['constant']['scale']

        # Gradients error function
        elif config['angular_scaling']['err_func_type'] == "gradient":
            scaling_params = config['angular_scaling']['gradient']
            self._aeff['Area_new'] = self._aeff['Area_new'] * (
                    1 + scaling_params['scale'] * gradient(theta,
                                                           scaling_params['range_min'],
                                                           scaling_params['range_max'])
            )

        # Step error function
        elif config['angular_scaling']['err_func_type'] == "step":
            scaling_params = config['angular_scaling']['step']
            break_points = list(zip(scaling_params['transition_pos'],
                                    scaling_params['transition_widths']))
            self._aeff['Area_new'] = self._aeff['Area_new'] * (
                    1 + scaling_params['scale'] * step(theta, break_points)
            )
        else:
            raise ValueError("Aeff angular scaling: unknown scaling function type '{:s}'"
                             .format(config['angular_scaling']['err_func_type']))
        # ------------------------------------------

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
        shutil.copy(db_file_path, db_file_path + '.bak')

        # Opening the database file
        db_file = pyfits.open(db_file_path)

        # Creating a new IRF table which will contain 4 more entries - new PSF/Aeff/Edisp/bkg.
        nrows_orig = len(db_file['CIF'].data)
        nrows_new = nrows_orig + 4
        hdu = pyfits.BinTableHDU.from_columns(db_file['CIF'].columns, nrows=nrows_new)

        # Aeff entry data
        aeff_vals = ['CTA', self.caldb_name, 'NONE', 'NONE', 'ONLINE',
                     'data/cta/{db:s}/bcf/{irf:s}'.format(db=self.caldb_name, irf=self.irf),
                     output_irf_file_name,
                     'BCF', 'DATA', 'EFF_AREA', 'NAME({:s})'.format(output_irf_name), 1,
                     '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA effective area']

        # PSF entry data
        psf_vals = ['CTA', self.caldb_name, 'NONE', 'NONE', 'ONLINE',
                    'data/cta/{db:s}/bcf/{irf:s}'.format(db=self.caldb_name, irf=self.irf),
                    output_irf_file_name,
                    'BCF', 'DATA', 'RPSF', 'NAME({:s})'.format(output_irf_name), 1,
                    '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA point spread function']

        # Edisp entry data
        edisp_vals = ['CTA', self.caldb_name, 'NONE', 'NONE', 'ONLINE',
                      'data/cta/{db:s}/bcf/{irf:s}'.format(db=self.caldb_name, irf=self.irf),
                      output_irf_file_name,
                      'BCF', 'DATA', 'EDISP', 'NAME({:s})'.format(output_irf_name), 1,
                      '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA energy dispersion']

        # Background entry data
        bkg_vals = ['CTA', self.caldb_name, 'NONE', 'NONE', 'ONLINE',
                    'data/cta/{db:s}/bcf/{irf:s}'.format(db=self.caldb_name, irf=self.irf),
                    output_irf_file_name,
                    'BCF', 'DATA', 'BKG', 'NAME({:s})'.format(output_irf_name), 1,
                    '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA background']

        # Filling the columns of the new table
        for col_i, colname in enumerate(hdu.columns.names):
            # First fill the previously existing data
            hdu.data[colname][:nrows_orig] = db_file['CIF'].data[colname]
            # Now fill the newly created entries
            hdu.data[colname][nrows_orig + 0] = aeff_vals[col_i]
            hdu.data[colname][nrows_orig + 1] = psf_vals[col_i]
            hdu.data[colname][nrows_orig + 2] = edisp_vals[col_i]
            hdu.data[colname][nrows_orig + 3] = bkg_vals[col_i]

        # Replacing the old IRF table
        db_file['CIF'].data = hdu.data

        # Saving the data base
        db_file.writeto(db_file_path, overwrite=True)
        db_file.close()

    def scale_irf(self, config):
        """
        This method performs scaling of the loaded IRF - both PSF and Aeff, if necessary.
        For the collection area two scalings can be applied: (1) vs energy and
        (2) vs off-axis angle. In both cases the scaling function is taken as
        (1 + scale * tanh((x-x0)/dx)). In case (1) the scaling value x is log10(energy).

        Parameters
        ----------
        config: dict
            A dictionary with the scaling settings. Must have following keys defined:
            "general", "aeff", "psf".

            Key "general" must be a dictionary, containing the following:
            caldb: str
                CALDB name, e.g. '1dc' or 'prod3b'.
            irf: str
                IRF name, e.g. 'South_z20_50h'
            output_irf_name: str
                The name of output IRF, say "my_irf".
            output_irf_file_name: str:
                The name of the output IRF file, e.g. 'irf_scaled_version.fits' (the name
                must follow the "irf_*.fits" template, "irf_scaled_version.fits"). The file
                will be put to the main directory of the chosen IRF.

            Keys "aeff" and "psf" must be dictionaries, containing the following:
            "energy_scaling": dict
                Contains setting for the energy scaling (see the structure below).
            "angular_scaling": dict
                Contains setting for the off-center angle scaling (see the structure below).

            In both cases, internally the above dictionaries should contain:
            "err_func_type": str
                The name of the scaling function to use. Accepted values are: "constant",
                "gradient" and "step".

            If err_func_type == "constant":
                scale: float
                    The scale factor. passing 1.0 results in no scaling.

            If err_func_type == "gradient":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                range_min: float
                    The x value (energy or off-center angle), that corresponds to -1 scale.
                range_max: float
                    The x value (energy or off-center angle), that corresponds to +1 scale.

            If err_func_type == "step":
                scale: float
                    The scale factor. passing 0.0 results in no scaling.
                transition_pos: list
                    The list of x values (energy or off-center angle), at which
                    step-like transitions occur. If scaling the energy dependence,
                    values must be in TeVs, if angular - in degrees.
                transition_widths: list
                    The list of step-like transition widths, that correspond to transition_pos.
                    For energy scaling the widths must be in log10 scale.

        Returns
        -------
        None

        """

        if self.am_ok:
            # Opening the IRF input file
            input_irf_file = pyfits.open(self.input_irf_file_name, 'readonly')

            # Scaling the PSF
            self._scale_psf(input_irf_file, config['psf'])

            # Scaling the Aeff
            self._scale_aeff(input_irf_file, config['aeff'])

            # Getting the new IRF and output file names
            # IRF name
            output_irf_name = config['general']['output_irf_name']
            # Output file name
            output_irf_file_name = config['general']['output_irf_file_name']

            # Figuring out the output path
            output_path = '{path:s}/data/cta/{caldb:s}/bcf/{irf:s}'.format(path=self.caldb_path,
                                                                           caldb=self.caldb_name,
                                                                           irf=self.irf)

            # Writing the scaled IRF
            input_irf_file.writeto(output_path + "/" + output_irf_file_name, overwrite=True)

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

        # Avoiding division by zero
        can_divide = self._aeff['Area'] > 0
        scale_map['Map'] = scipy.zeros_like(self._aeff['Area_new'])
        scale_map['Map'][can_divide] = self._aeff['Area_new'][can_divide] / self._aeff['Area'][can_divide]

        wh_nan = scipy.where(scipy.isnan(scale_map['Map']))
        scale_map['Map'][wh_nan] = 0
        scale_map['Map'] -= 1

        return scale_map

    def get_psf_scale_map(self):
        """
        This method returns the PSF scale map, which can be useful for check of the used settings.
        Must be run after the scale_irf() method.

        Returns
        -------
        dict:
            A dictionary with the PSF scale map.
        """

        scale_map = dict()

        scale_map['E_edges'] = scipy.concatenate((self._psf['Elow'], [self._psf['Ehigh'][-1]]))
        scale_map['Theta_edges'] = scipy.concatenate((self._psf['ThetaLow'], [self._psf['ThetaHi'][-1]]))

        # Find all "sigma" values - tells how many PSF components we have in the IRF file
        column_names = self._psf.keys()
        sigma_columns = list(filter(lambda s: ("sigma" in s.lower()) and not ("new" in s.lower()),
                                    column_names))

        for sigma_column in sigma_columns:
            # Avoiding division by zero
            can_divide = self._psf[sigma_column] > 0

            scale_map[sigma_column] = scipy.zeros_like(self._psf[sigma_column])
            scale_map[sigma_column][can_divide] = self._psf[sigma_column + '_new'][can_divide] / self._psf[sigma_column][can_divide]

            wh_nan = scipy.where(scipy.isnan(scale_map[sigma_column]))
            scale_map[sigma_column][wh_nan] = 0
            scale_map[sigma_column] -= 1

        return scale_map

    def plot_aeff_scale_map(self, vmin=-0.5, vmax=0.5):
        """
        This method plots the collection area scale map, which can be useful
        for checking the used settings.
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

        pyplot.title("Collection area scale map")
        pyplot.semilogx()

        pyplot.xlabel('Energy, TeV')
        pyplot.ylabel('Off-center angle, deg')
        pyplot.pcolormesh(scale_map['E_edges'], scale_map['Theta_edges'], scale_map['Map'].transpose(),
                          cmap='bwr', vmin=vmin, vmax=vmax)
        pyplot.colorbar()

    def plot_psf_scale_map(self, vmin=-0.5, vmax=0.5):
        """
        This method plots the PSF scale map (for sigma_1 parameter), which can be useful
        for checking the used settings.
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

        scale_map = self.get_psf_scale_map()

        pyplot.title("PSF $\sigma_1$ scale map")
        pyplot.semilogx()

        pyplot.xlabel('Energy, TeV')
        pyplot.ylabel('Off-center angle, deg')
        pyplot.pcolormesh(scale_map['E_edges'], scale_map['Theta_edges'], scale_map['sigma_1'].transpose(),
                          cmap='bwr', vmin=vmin, vmax=vmax)
        pyplot.colorbar()
