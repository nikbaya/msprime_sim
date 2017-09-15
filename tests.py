#!/usr/bin/env python

from __future__ import division
import msprime
import unittest
import msprime_sim as msim
import numpy.testing as npt
import numpy as np
import time, sys, traceback, argparse
import src.phenotypes as ph
import src.printing as pr
import tqdm

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.h2_A = 0.3
args.h2_D = 0.1
args.h2_AC = 0.2
args.p_causal = 1
args.n = 20
args.m = 1000000
args.maf = 0.05
args.n_chr = 1
args.dominance = False
args.gxe = False
args.geno_prop = None
args.same_causal_sites = False
args.case_control = False
args.prevalence = 0.1
args.n_cases = 1000
args.sample_prevalence = None
args.include_pop_strat = False
args.s2 = 0.1
args.write_pheno = False
args.progress_bars = False
args.debug = True
args.dominance = False
args.gxe = False
args.include_pop_strat = False

log = pr.Logger('testing.log')

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

    def loadsim(self, tree_path):
        test_tree = msprime.load(tree_path)
        tree_sequence_list = []
        tree_sequence_list.append(test_tree)
        m_total = tree_sequence_list[0].get_num_mutations()
        return m_total, tree_sequence_list

    def test_pheno(self):
        global args
        C = np.zeros(args.n)
        # DEV: THESE TESTS WILL FAIL NOW, NEED TO REPLACE HARDCODED PHENOS.
        # Hard-coded result from just additive.
        y_add = np.array([ 0.10124291, -0.02349692, 1.32977501, -1.94930528, -0.04203848,
                  -0.59203988, -0.18556655, -1.40978275, 0.34049718, 0.50990399,
                  0.03416461, -0.26384922, -0.40792179, -0.80574391, 1.84488544,
                  2.29662925, -0.36562796, -0.02769463, -1.166248, 0.782217])

        y_add_dom = np.array([-0.46232299, 0.42799796, 1.30738915, -1.44502127, 0.54228883,
                    1.46883349, 1.06715295, -0.90632239, 0.57748057, -1.87934203,
                    -0.73929674, 0.84053437, 1.15491044, -0.9038878,  -0.4664165,
                    -1.43336406,  0.88217141, -0.43847431, -0.42952555, 0.83521447])

        y_add_dom_gxe = np.array([-0.04593424, 0.21331692, 0.25270979, 0.58572902, 2.72193019,
            0.11658301, 0.01357718, -1.62950823, -1.75267563, 1.52283816,
            0.35947773, -0.76093934, -0.49633856, -0.23657767, -0.90878785,
            0.05436258, 1.18573913, -0.44498875, 0.04397375, -0.79448717])

        # Read in a test tree.
        m_total, tree_sequence_list = self.loadsim('test_data/test_sequence_add.tree')
        
        if args.dominance is not True: args.h2_D = 0
        if args.gxe is not True: args.h2_AC = 0
        if args.include_pop_strat is not True: args.s2 = 0

        # args.debug is turned on - so this should give the same results.
        y = ph.get_phenotypes(args, args.n, 1, tree_sequence_list, m_total, log)

        self.assertTrue(np.max(np.abs(y - y_add)) < 1e-5)

        args.dominance = True
        args.h2_D = 0.1

        m_total, tree_sequence_list = self.loadsim('test_data/test_sequence_add_dom.tree')
        y = ph.get_phenotypes(args, args.n, 1, tree_sequence_list, m_total, log)

        self.assertTrue(np.max(np.abs(y - y_add_dom)) < 1e-5)

        args.gxe = True
        args.h2_AC = 0.2

        m_total, tree_sequence_list = self.loadsim('test_data/test_sequence_add_dom_gxe.tree')
        y = ph.get_phenotypes(args, args.n, 1, tree_sequence_list, m_total, log)

        self.assertTrue(np.max(np.abs(y - y_add_dom_gxe)) < 1e-5)

        # want to ensure that this is true for all possible combinations of heritability contributions.
        

if __name__ == '__main__':
    unittest.main()