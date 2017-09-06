#!/usr/bin/env python

from __future__ import division
import unittest
import msprime_sim as msim
import numpy.testing as npt
import numpy as np
import time, sys, traceback, argparse

class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def loadData(self, bfile, phifile, mafMin=0.05):
        snp_file, snp_obj = bfile + '.bim', ps.PlinkBIMFile
        ind_file, ind_obj = bfile + '.fam', ps.PlinkFAMFile
        array_file, array_obj = bfile + '.bed', fb.PlinkBEDFile

        array_snps = snp_obj(snp_file)

        m = len(array_snps.IDList)
        print('Read list of {m} test SNPs from {f}'.format(m=m, f=snp_file))

        # Read fam.
        array_indivs = ind_obj(ind_file)
        n = len(array_indivs.IDList)
        print('Read list of {n} individuals from {f}'.format(n=n, f=ind_file))

        # Read genotype array.
        print('Reading genotypes from {fname}'.format(fname=array_file))
        geno_array = array_obj(array_file, n, array_snps, keep_snps=None,
        keep_indivs=None, mafMin=mafMin)

        n_pass_y_sims = 1
        phi_cols = range(2,n_pass_y_sims+2)
        phi_names = [''.join(str(i) for i in z) for z in zip(np.tile('y', n_pass_y_sims), range(n_pass_y_sims))]
        PhenoFile = ps.__ID_List_Factory__(['IID', 'FID'] + phi_names, 0, '.phi',
            usecols=[0, 1] + phi_cols)
        pheno_file, pheno_obj = phifile, PhenoFile
        phi = pheno_obj(pheno_file)
        phi = np.array(phi.df[phi_names])

        return geno_array, array_indivs, phi

    def test_silly(self):
    	self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()