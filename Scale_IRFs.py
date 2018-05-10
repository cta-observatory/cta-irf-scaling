#!/usr/bin/env python
# coding=utf-8

import os
import re
import glob
import pyfits
import scipy

from matplotlib import pyplot


class CalDB:
    def __init__(self, caldb_name, irf_name, verbose=False):
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
        for psf_i in range(0, n_psf_components):
            input_irf_file['POINT SPREAD FUNCTION'].data['SIGMA_{:d}'.format(psf_i+1)] *= psf_scale
            
    def _scale_aeff(self, input_irf_file,
                    aeff_scale_energy, aeff_norm_energy, aeff_transition_width_energy,
                    aeff_scale_theta, aeff_norm_theta, aeff_transition_width_theta):
        
        self._aeff['Elow']     = input_irf_file['Effective area'].data['Energ_lo'][0]
        self._aeff['Ehigh']    = input_irf_file['Effective area'].data['Energ_hi'][0]
        self._aeff['ThetaLow'] = input_irf_file['Effective area'].data['Theta_lo'][0]
        self._aeff['ThetaHi']  = input_irf_file['Effective area'].data['Theta_hi'][0]
        self._aeff['Area']     = input_irf_file['Effective area'].data['EffArea'][0].transpose()
        self._aeff['E']     = scipy.sqrt(self._aeff['Elow']*self._aeff['Ehigh'])
        self._aeff['Theta'] = (self._aeff['ThetaLow']+self._aeff['ThetaHi'])/2.0

        energy, theta = scipy.meshgrid(self._aeff['E'], self._aeff['Theta'], indexing='ij')

        scaling_val = scipy.log10(energy / aeff_norm_energy) / aeff_transition_width_energy
        self._aeff['Area_new'] = self._aeff['Area'] * (1 + aeff_scale_energy * scipy.tanh(scaling_val))

        scaling_val = (theta - aeff_norm_theta) / aeff_transition_width_theta
        self._aeff['Area_new'] = self._aeff['Area_new'] * (1 + aeff_scale_theta * scipy.tanh(scaling_val))
        
        input_irf_file['Effective area'].data['EffArea'][0] = self._aeff['Area_new'].transpose()

    def _append_irf_to_db(self, output_irf_name, output_irf_file_name):
        db_file_path = '{path:s}/data/cta/{caldb:s}/caldb.indx'.format(path=self.caldb_path, caldb=self.caldb_name)
        db_file = pyfits.open(db_file_path)

        nrows_orig = len(db_file['CIF'].data)
        nrows_new = nrows_orig + 4
        hdu = pyfits.BinTableHDU.from_columns(db_file['CIF'].columns, nrows=nrows_new)

        aeff_vals = ['CTA', '1DC', 'NONE', 'NONE', 'ONLINE', 'data/cta/1dc/bcf/' + self.irf, output_irf_file_name,
                     'BCF', 'DATA', 'EFF_AREA', 'NAME({:s})'.format(self.irf + '_' + output_irf_name), 1,
                     '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA effective area']

        psf_vals = ['CTA', '1DC', 'NONE', 'NONE', 'ONLINE', 'data/cta/1dc/bcf/' + self.irf, output_irf_file_name,
                    'BCF', 'DATA', 'RPSF', 'NAME({:s})'.format(self.irf + '_' + output_irf_name), 1,
                    '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA point spread function']

        edisp_vals = ['CTA', '1DC', 'NONE', 'NONE', 'ONLINE', 'data/cta/1dc/bcf/' + self.irf, output_irf_file_name,
                      'BCF', 'DATA', 'EDISP', 'NAME({:s})'.format(self.irf + '_' + output_irf_name), 1,
                      '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA energy dispersion']

        bkg_vals = ['CTA', '1DC', 'NONE', 'NONE', 'ONLINE', 'data/cta/1dc/bcf/'+self.irf, output_irf_file_name,
                    'BCF', 'DATA', 'BKG', 'NAME({:s})'.format(self.irf + '_' + output_irf_name), 1,
                    '2014-01-30', '00:00:00', 51544.0, 0, '14/01/30', 'CTA background']

        for col_i, colname in enumerate(hdu.columns.names):
            hdu.data[colname][:nrows_orig] = db_file['CIF'].data[colname]
            hdu.data[colname][nrows_orig] = aeff_vals[col_i]
            hdu.data[colname][nrows_orig+1] = psf_vals[col_i]
            hdu.data[colname][nrows_orig+2] = edisp_vals[col_i]
            hdu.data[colname][nrows_orig+3] = bkg_vals[col_i]

        db_file['CIF'].data = hdu.data

        db_file.writeto(db_file_path, clobber=True)

    def scale_irf(self,
                  psf_scale=1,
                  aeff_scale_energy=0.0, aeff_norm_energy=1.0, aeff_transition_width_energy=1.0,
                  aeff_scale_theta=0.0, aeff_norm_theta=1.0, aeff_transition_width_theta=1.0,
                  output_irf_file_name=""):
        
        if not self.am_ok:
            print("ERROR: something's wrong with the CALDB/IRF names.")

        input_irf_file = pyfits.open(self.input_irf_file_name, 'readonly')
        self._scale_psf(input_irf_file, psf_scale)
        self._scale_aeff(input_irf_file,
                         aeff_scale_energy, aeff_norm_energy, aeff_transition_width_energy,
                         aeff_scale_theta, aeff_norm_theta, aeff_transition_width_theta)

        if output_irf_file_name == "":
            output_psf_part = "P-{:.1f}".format(psf_scale)
            output_aeff_energy_part = "A-{:.1f}-{:.1f}-{:.1f}".format(aeff_scale_energy,
                                                                               aeff_norm_energy,
                                                                               aeff_transition_width_energy)
            output_aeff_theta_part = "{:.1f}-{:.1f}-{:.1f}".format(aeff_scale_theta,
                                                                               aeff_norm_theta,
                                                                               aeff_transition_width_theta)
            output_irf_name = output_psf_part + "_" + output_aeff_energy_part + output_aeff_theta_part
            output_irf_file_name = "irf_{:s}.fits".format(output_irf_name)

        output_path = '{path:s}/data/cta/{caldb:s}/bcf/{irf:s}'.format(path=self.caldb_path,
                                                                       caldb=self.caldb_name,
                                                                       irf=self.irf)

        input_irf_file.writeto(output_path + "/" + output_irf_file_name, clobber=True)

        self._append_irf_to_db(output_irf_name, output_irf_file_name)

    def get_aeff_scale_map(self):
        scale_map = dict()
        
        scale_map['E_edges'] = scipy.concatenate((self._aeff['Elow'], [self._aeff['Ehigh'][-1]]))
        scale_map['Theta_edges'] = scipy.concatenate((self._aeff['ThetaLow'], [self._aeff['ThetaHi'][-1]]))

        scale_map['Map'] = self._aeff['Area_new'] / self._aeff['Area']
        wh_nan = scipy.where(scipy.isnan(scale_map['Map']))
        scale_map['Map'][wh_nan] = 0
        scale_map['Map'] -= 1
        
        return scale_map
        

# =================
# === Main code ===
# =================

caldb_path = os.environ['CALDB']
caldb = '1dc'
irf = 'North_z20_50h'

psf_scale = 2.0
aeff_norm_energy = 1.0
aeff_norm_theta = 2.5
aeff_transition_width_energy = 1
aeff_transition_width_theta = 1.0
aeff_scale_energy = +0.2
aeff_scale_theta = -0.2

caldb = CalDB(caldb, irf, verbose=False)
caldb.scale_irf(psf_scale,
                aeff_scale_energy, aeff_norm_energy, aeff_transition_width_energy,
                aeff_scale_theta, aeff_norm_theta, aeff_transition_width_theta)

pyplot.clf()
pyplot.semilogx()

pyplot.xlabel('Energy, TeV')
pyplot.ylabel('Off-center angle, deg')
pyplot.pcolormesh(aeff['E_edges'], aeff['Theta_edges'], aeff['Relative_map'].transpose(), 
                  cmap='bwr', vmin=-0.5, vmax=0.5)
pyplot.colorbar()

pyplot.show()
