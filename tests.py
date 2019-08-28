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
import src.tree_sequence as ts
import src.ldscores as ld
import tqdm
import random

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

    # DEV: Want to do these same phenotype checks when p_causal is set.
    def test_pheno(self):
        global args

        if args.dominance is not True: args.h2_D = 0
        if args.gxe is not True: args.h2_AC = 0
        if args.include_pop_strat is not True: args.s2 = 0
        
        # Hard-coded results (first 5 phenotypes).
        yA_hc = np.array([ 0.68553785, -0.31882745, 2.05762415, -0.72919558, 0.54672029])
        yAD_hc = np.array([-0.76057419, -0.62355073, 0.71763764, 1.26129078, 1.47807986])
        yADGxE_hc = np.array([ 0.68380307, 0.20878173, 1.7423659, 1.22472338, -0.02755698])
        yADGxEs_hc = np.array([1.55558875, 0.9081014, -0.12515331, -0.47767391, 0.4529098])
        # Read in a test tree.
        m_total, tree_sequence_list = self.loadsim('test_data/test_tree.chr1.sim1.tree')
        
        # args.debug is turned on - so this should give the same results.
        yA, C = ph.get_phenotypes(args, args.n, 1, tree_sequence_list, m_total, log)
        self.assertTrue(np.max(np.abs(yA_hc - yA[:5])) < 1e-5)

        args.dominance = True
        args.h2_D = 0.1

        # Now test inclusion of dominance matches.
        yAD, C = ph.get_phenotypes(args, args.n, 1, tree_sequence_list, m_total, log)
        self.assertTrue(np.max(np.abs(yAD_hc - yAD[:5])) < 1e-5)
        
        args.gxe = True
        args.h2_AC = 0.2

        # Now test inclusion of dominance and gxe matches.
        yADGxE, C = ph.get_phenotypes(args, args.n, 1, tree_sequence_list, m_total, log)
        self.assertTrue(np.max(np.abs(yADGxE_hc - yADGxE[:5])) < 1e-5)

        args.include_pop_strat = True
        args.s2 = 0.1
        # Now test inclusion of population stratification.
        yADGxEs, C = ph.get_phenotypes(args, args.n, 1, tree_sequence_list, m_total, log)
        self.assertTrue(np.max(np.abs(yADGxEs_hc - yADGxEs[:5])) < 1e-5)    

    # Want to also use set mutations to check p_causal.
    def test_set_mutations_in_tree(self):
        m_total, tree_sequence_list = self.loadsim('test_data/test_tree.chr1.sim1.tree')
        
        # Restrict to subset of SNPs at random. This is stochastic.
        prop = 0.1
        tree_sequence, m_causal, restricted_index = ts.set_mutations_in_tree(tree_sequence_list[0], prop)
        prop_emp = tree_sequence.get_num_sites() / tree_sequence_list[0].get_num_sites()
        acc = prop / 10
        self.assertTrue(np.abs(prop_emp - prop) < acc)
        
        prop = 0.5
        tree_sequence, m_causal, restricted_index = ts.set_mutations_in_tree(tree_sequence_list[0], prop)
        prop_emp = tree_sequence.get_num_sites() / tree_sequence_list[0].get_num_sites()
        acc = prop / 10
        self.assertTrue(np.abs(prop_emp - prop) < acc)

    def test_ld_scores(self):

        global args

        lNA_hc = np.array([10.05235442, 12.46504893, 7.62187527, 15.73019434, 8.11817962])
        lND_hc = np.array([ 6.47247612,  6.25967469, 2.65875914, 10.45212777, 2.93538821])

        args.load_tree_sequence = 'test_data/test_tree.chr1.sim1.tree'
        tree_sequence_list, tree_sequence_list_geno, m, m_start, m_total, m_geno, m_geno_start, m_geno_total, N, n_pops = ts.load_tree_sequence(args, log)
        
        # When there's no sampling.
        args.ldscore_sampling_prop = None
        ldsc_index = None
        n_ldsc = N

        sim = 1
        args.ld_wind_snps = 100
        args.chunk_size = 50
        args.write_l2 = False
        lN_A, lN_D = ld.get_ldscores(args, m_geno, m_geno_start, m_geno_total, tree_sequence_list_geno, n_ldsc, ldsc_index, sim, log)

        self.assertTrue(np.max(np.abs(lN_A[:5] - lNA_hc)) < 1e-5)
        self.assertTrue(np.max(np.abs(lN_D[:5] - lND_hc)) < 1e-5)

        # When we sample 50%.
        lNA_hc = np.array([10.95534412, 11.4322989,  7.79579152, 17.17340013, 9.33888899])
        lND_hc = np.array([ 6.57092924,  5.84489464, 2.58647217, 10.77842306, 3.75377223])

        # DEV: Note that this flag can result in errors if the maf is low and the 
        # sampling proportion is low - it'll result in a division by 0 as the variance is 0.
        args.ldscore_sampling_prop = 0.95

        log.log('Using a subset of individuals from the sampled tree to determine LD scores - LD score sampling proportion: {ld}'.format(ld=args.ldscore_sampling_prop))

        if args.debug:
            random.seed(1)
            np.random.seed(1)

        n_ldsc = int(N*args.ldscore_sampling_prop)
        ldsc_index = random.sample(xrange(N), n_ldsc)

        lN_A, lN_D = ld.get_ldscores(args, m_geno, m_geno_start, m_geno_total, tree_sequence_list_geno, n_ldsc, ldsc_index, sim, log)
        self.assertTrue(np.max(np.abs(lN_A[:5] - lNA_hc)) < 1e-5)
        self.assertTrue(np.max(np.abs(lN_D[:5] - lND_hc)) < 1e-5)

    def test_ldsc_pcgc_h2_estimate(self):
        # This is a poor test that just diagnoses if there's a problem somewhere in the code.
        # Doesn't give any further information.
        global args
        args.ldscore_sampling_prop = None
        args.ld_wind_snps = 1000
        args.chunk_size = 50
        args.write_l2 = False

        args.pcgc = True
        args.free_and_no_intercept = True
        args.no_intercept = False
        args.rec_map_chr = None
        args.rec_map = None
        args.n_sims = 1
        args.ldsc = True
        args.ldscore_within_sample = False
        args.write_trees = False
        args.n_blocks = 200

        # Run the whole thing, loading from a dumped tree...and check it matches
        # the hard coded estimates.
        ldsc_h2_hc = np.array([-0.43536562,  1.17030242,  0.26852651,  0.9170707, -0.94560754,  1.11171601])
        ldsc_int_hc = np.array([0.00208552,  1, -0.28399289,  1, -0.65708724,  1])
        pcgc_hc = np.array([0.90692179, -0.75257328, -0.96656856])

        h2_ldsc, h2_pcgc, h2_ldsc_int = msim.simulate_tree_and_betas(args, log)

        h2_ldsc_check = np.array([h2_ldsc['h2_A'][0], h2_ldsc['int_A'][0],
            h2_ldsc['h2_D'][0], h2_ldsc['int_D'][0], 
            h2_ldsc['h2_AC'][0], h2_ldsc['int_AC'][0]])
        h2_ldsc_int_check = np.array([h2_ldsc_int['h2_A'][0], h2_ldsc_int['int_A'][0],
            h2_ldsc_int['h2_D'][0], h2_ldsc_int['int_D'][0], 
            h2_ldsc_int['h2_AC'][0], h2_ldsc_int['int_AC'][0]])
        pcgc_check = np.array([h2_pcgc['h2_A'][0], h2_pcgc['h2_D'][0], h2_pcgc['h2_AC'][0]])
        
        self.assertTrue(np.max(pcgc_check - pcgc_hc) < 1e-5)
        self.assertTrue(np.max(h2_ldsc_check - ldsc_h2_hc) < 1e-5)
        self.assertTrue(np.max(h2_ldsc_int_check - ldsc_int_hc) < 1e-5)

if __name__ == '__main__':
    unittest.main()

# DEV: Things to improve...
    # Currently the method of saving things to disk is stupid - fix this.
        # Get this done today.
    # Add Danfeng's stuff - writing the causal effect to disk.
    # Add this R * beta thing.
        # Option to write it to disk.
    # Organise the options into some kind of useful order.
        # Check how I've done this in the README.

    # Fixing potential errors that can result from running ldscore-sampling-proportion.
    # Come up with tests for the case-control and ascertainment stuff.

# DEV: things to test.
# Check that maf is set at a value, no SNPs survive that have maf > than that value.
# Check that when the geno_prop is set to a value, that approximately that
# proportion of SNPs are genotyped.

# Check prevalence in the sample and in the population when prevalence is used.
# Check the sample prevalence.

# Check n_cases when that flag is used.
# Check that when ldscore-sampling-prop is set to 1, that it matches up with the
# usual LD score without that flag.
