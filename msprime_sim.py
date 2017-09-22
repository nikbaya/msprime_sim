#!/usr/bin/env python

from __future__ import division
import msprime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import math
import time, sys, traceback, argparse
import pandas as pd
import src.printing as pr
import src.msprime_sim_scenarios as mssim
# import src.sumstats as sumstats
import src.regressions as reg
import statsmodels.api as sm
import tqdm
import tempfile
import scipy.stats as sp
import random
from itertools import islice

def p_to_z(p, N):
	# Convert p-value and N to standardized beta.
	return np.sqrt(sp.chi2.isf(p, 1))

def progress(progress_bars, range_object, total):
	if progress_bars:
		return tqdm.tqdm(range_object, total=total)
	else:
		return range_object

def sub_chr(s, chr):
	# Substitute chr for @, else append chr to the end of str.
	if '@' not in s:
		s += '@'
	return s.replace('@', str(chr))

def l2(x, n):
	sq = np.square(x)
	return sq - 1/n

def write_trees(out, tree_sequence, chr, m, n_pops, N, sim, vcf, sample_index):
	# DEV: Throw a warning if you try to do this and n_sims is high.
	vcf_name = out + ".vcf"
	with open(vcf_name, "w") as vcf_file:
		tree_sequence.write_vcf(vcf_file, ploidy=2)

	print N
	# DEV: update the function - no longer require N (double check this).
	N = int(tree_sequence.get_sample_size() / 2)
	print N
	
	# Create altered IDs an Family IDs to replace the .fam file that we will create.
	fam_id = np.tile('msp', N)
	index_old = [i for i in xrange(N)]
	# Have to change the '0' ID to something else, as plink doesn't like IIDs to be '0'.
	index_old[0] = 'A'
	# Similarly for the new list of indices, plink doesn't like IIDs to be '0'.
	index_new = [i for i in sample_index]
	matches = [x for x in index_new if x == 0]
	if len(matches) == 1:
		index_new[index_new.index(0)] = 'A'

	# Create a new table to define the re-indexing of the tree. 
	# Writing to .vcf does not save the sample numbers, so we need to keep track of these and 
	# replace them in the .fam file.

	d={'old_fam':fam_id, 'old_within_fam':index_new, 'new_fam':fam_id, 'new_within_fam':index_old}
	df=pd.DataFrame(d)
	tmp_index_tsv = out + '.index.tmp.tsv'
	df.to_csv(tmp_index_tsv, sep='\t', header=False, index=False)

	if vcf is False:
		# Note that the following line is OS dependent. OSX requires a gap after '-i'.
		os.system("sed -i.bak '1,/msp_0/ s/msp_0/msp_A/' " + vcf_name)
		# Convert to Plink bed format - need to ensure that plink is in your path.
		bfile_out = out + ".chr" + str(chr+1) + ".sim" + str(sim+1)
		os.system("../plink/plink --vcf " + vcf_name + " --out " + bfile_out + " --make-bed")
		# Now, fix the chromosome number and the names of the mutations.
		mut_names=np.core.defchararray.add('rs.' + str(chr+1) + ".", np.arange(1,m+1).astype('str'))
		chr_vec=np.tile(chr+1, m)
		d={'chr':chr_vec, 'rs': mut_names}
		df=pd.DataFrame(d)
		tmp_tsv = out + '.tmp.tsv'
		tmp_bim = out + '.bim_tmp.tsv'
		df.to_csv(tmp_tsv, sep='\t', header=False, index=False)
		os.system('cut -f 3,4,5,6 ' +  bfile_out + '.bim > ' + tmp_bim)
		os.system('paste ' + tmp_tsv + ' ' + tmp_bim + ' > ' + bfile_out + '.bim')
		os.system('rm ' + tmp_tsv + '; rm ' + tmp_bim)

		# Now remove the .vcf files.
		os.system('rm ' + vcf_name + '; rm ' + vcf_name + '.bak; rm ' + bfile_out + '.fam.bak')

		os.system('../plink/plink --update-ids ' + tmp_index_tsv + ' --bfile ' + bfile_out + ' --make-bed --out ' + bfile_out)
		# Rename 'A' to '0'.
		os.system("sed -i.bak 's/msp A/msp 0/' " + bfile_out + '.fam')
		# os.system('rm ' + tmp_index_tsv)
		# Remove the .bak and temporary files
		os.system('rm ' + bfile_out + '*.bak')
		os.system('rm ' + bfile_out + '*~')

	pop_ann = np.empty(N)

	for pops in xrange(n_pops):
		pop_leaves = tree_sequence.get_samples(population_id=pops)
		pop_ann[map(int, [x/2 for x in pop_leaves[0::2]])] = pops

	if chr==0:
		df_pop=pd.DataFrame({'sample':sample_index, 'population':pop_ann.astype(int)})
		df_pop.to_csv(out + ".sim" + str(sim+1) + '.pop.tsv', sep='\t', header=True, index=False)

def getBlockLefts(coords, max_dist):
	M = len(coords)
	j = 0
	block_left = np.zeros(M)
	for i in xrange(M):
		while j < M and abs(coords[j] - coords[i]) > max_dist:
			j += 1
		block_left[i] = j

	return block_left

# Normalize additive and dominance entries in sample.
def standardize(X):
	n = len(X)
	mean_X = np.mean(X)
	p = mean_X / 2
	X_A = (X - mean_X) / np.std(X)

	X_D = np.copy(X)
	X_D[X_D == 2] = 0

	G = np.concatenate((np.ones((n, 1)), X_A[:, None], X_D[:, None]), axis = 1)
	X_D = math.sqrt(n) * np.linalg.qr(G)[0][:,-1]

	# X_D = np.ones(n)
	# X_D[X == 0] = - p / (1 - p)
	# X_D[X == 2] = - (1 - p) / p
	# X_D = (X_D - np.mean(X_D)) / np.std(X_D)
	
	return X_A, X_D

def nextSNP(variant, Xs, index=None):
	if index is None:
		var_tmp = np.array(map(int, variant.genotypes[0::2])) + np.array(map(int, variant.genotypes[1::2]))
	else:
		var_tmp = np.array(map(int, variant.genotypes[0::2][index])) + np.array(map(int, variant.genotypes[1::2][index]))
	
	n = len(var_tmp)
	X_A, X_D = standardize(var_tmp)

	X_E = np.zeros(n)

	if Xs is not None:
		Xs_A, Xs_D = standardize(Xs)
		G = np.concatenate((np.ones((n, 1)), X_A[:, None], Xs_A[:, None], X_D[:, None], Xs_D[:, None], np.multiply(X_A, Xs_A)[:, None]), axis=1)

		X_E = math.sqrt(n) * np.linalg.qr(G)[0][:, -1]

	return X_A, X_D, X_E

def nextSNP_add(variant, Xs, index=None):
	X_A, X_D, X_E = nextSNP(variant, Xs, index=None)
	return X_A

def nextSNP_dom(variant, Xs, index=None):
	X_A, X_D, X_E = nextSNP(variant, Xs, index=None)
	return X_D

def nextSNP_epi(variant, Xs, index=None):
	X_A, X_D, X_E = nextSNP(variant, Xs, index=None)
	return X_E

# def obtain_K(variants, K_A, K_D, K_AC, C, c, m, n, progress_bars, index):
# 	# Number of chunks of length c in m sites.
# 	n_c = np.floor(m/c).astype(int)
# 	X_A, X_D, X_E = np.empty((n, c)), np.empty((n, c)), np.empty((n, c))
# 	C_mat = np.repeat(C, c).reshape((n, c))

# 	for i in progress(progress_bars, xrange(n_c), total=n_c):
# 		k = 0

# 		while k < c:
# 			variant = variants.next()
# 			X_A[:,k], X_D[:,k], X_E[:,k] = nextSNP(variant, index)
# 			k += 1
		
# 		K_A += np.dot(X_A, X_A.T)
# 		K_D += np.dot(X_D, X_D.T)
# 		K_E += np.dot(X_E, X_E.T)
# 		K_AC += np.dot(C_mat * X_A, (C_mat * X_A).T)
		
# 	# The final chunk.
# 	if (n_c * c) < m:
# 		k = 0
# 		c = m - (n_c * c)
# 		X_A, X_D, X_E = np.empty((n, c)), np.empty((n, c)), np.empty((n, c))
# 		C_mat = np.repeat(C, c).reshape((n, c))

# 		while k < c:
# 			variant = variants.next()
# 			X_A[:,k], X_D[:,k], X_E[:,k] = nextSNP(variant, index)
# 			k += 1

# 		K_A += np.dot(X_A, X_A.T)
# 		K_D += np.dot(X_D, X_D.T)
# 		K_E += np.dot(X_E, X_E.T)
# 		K_AC += np.dot(C_mat * X_A, (C_mat * X_A).T)

# 	return K_A, K_D, K_E, K_AC

# m := number of SNPs
# N := number of individuals
def corSumVarBlocks(variants, block_left, c, m, N, Xs, snp_getter1, snp_getter2, both, ldsc_index):
	track_A = []
	track_B = []
	block_sizes = np.array(np.arange(m) - block_left)
	block_sizes = (np.ceil(block_sizes / c) * c).astype(int)
	cor_sum1, cor_sum2 = np.zeros(m), np.zeros(m)

	# b = index of first SNP for which SNP 0 is not included in LD Score
	b = np.nonzero(block_left > 0)
	# print b
	if np.any(b):
		b = b[0][0]
	else:
		b = m
	b = int(np.ceil(b/c)*c)  # Round up to a multiple of c

	if b > m:
		c = 1
		b = m
	l_A = 0  # l_A := index of leftmost SNP in matrix A

	k = 0
	
	A1, A2 = np.empty((N,b)), np.empty((N,b))
	while k < b:
		variant = variants.next()
		
		A1[:,k] = snp_getter1(variant, Xs, index=ldsc_index)
		# print variant.index
		A2[:,k] = snp_getter2(variant, Xs, index=ldsc_index)
		# print variant.index
		k += 1

	rfuncAB = np.zeros((b, c))
	rfuncBA = np.zeros((c, b))
	rfuncBB = np.zeros((c, c))

	# Chunk inside of block
	for l_B in xrange(0, b, c):  # l_B := index of leftmost SNP in matrix B
		B1 = A1[:, l_B:l_B+c]
		B2 = A2[:, l_B:l_B+c]
		np.dot(A1.T, B2 / N, out=rfuncAB)
		rfuncAB = l2(rfuncAB, N)
		cor_sum1[l_A:l_A+b] += np.sum(rfuncAB, axis=1)
		if both: cor_sum2[l_B:l_B+c] += np.sum(rfuncAB, axis=0)

	# Chunk to right of block
	b0 = int(b)
	md = int(c*np.floor(m/c))
	end = md + 1 if md != m else md

	for l_B in xrange(b0, end, c):
		# Update the block
		old_b = b

		b = block_sizes[l_B]
		if l_B > b0 and b > 0:
			# block_size can't increase more than c
			# block_size can't be less than c unless it is zero
			# Both of these things make sense
			A1 = np.hstack((A1[:, old_b-b+c:old_b], B1))
			A2 = np.hstack((A2[:, old_b-b+c:old_b], B2))
			l_A += old_b-b+c
		elif l_B == b0 and b > 0:
			A1 = A1[:, b0-b:b0]
			A2 = A2[:, b0-b:b0]
			l_A = b0-b
		elif b == 0:  # no SNPs to left in window, e.g., after a sequence gap
			A1 = np.array(()).reshape((N, 0))
			A2 = np.array(()).reshape((N, 0))
			l_A = l_B
		if l_B == md:
			c = m - md
			rfuncAB = np.zeros((b, c))
			rfuncBA = np.zeros((c, b))
			rfuncBB = np.zeros((c, c))
		if b != old_b:
			rfuncAB = np.zeros((b, c))
			rfuncBA = np.zeros((c, b))

		k = 0
		B1, B2 = np.empty((N,c)), np.empty((N,c))
		while k < c:
			variant = variants.next()
			B1[:,k] = snp_getter1(variant, Xs, index=ldsc_index)
			B2[:,k] = snp_getter2(variant, Xs, index=ldsc_index)
			k += 1

		np.dot(A1.T, B2 / N, out=rfuncAB)
		rfuncAB = l2(rfuncAB, N)
		np.dot(B1.T, A2 / N, out=rfuncBA)
		rfuncBA = l2(rfuncBA, N)
		np.dot(B1.T, B2 / N, out=rfuncBB)
		rfuncBB = l2(rfuncBB, N)

		cor_sum1[l_A:l_A+b] += np.sum(rfuncAB, axis=1)
		cor_sum1[l_B:l_B+c] += np.sum(rfuncBA, axis=1)
		cor_sum1[l_B:l_B+c] += np.sum(rfuncBB, axis=1)

		if both:
			cor_sum2[l_B:l_B+c] += np.sum(rfuncAB, axis=0)
			cor_sum2[l_A:l_A+b] += np.sum(rfuncBA, axis=0)
			cor_sum2[l_B:l_B+c] += np.sum(rfuncBB, axis=0)     

	return (cor_sum1, cor_sum2) if both else cor_sum1

def set_mutations_in_tree(tree_sequence, p_causal):
	causal_sites = []
	for site in tree_sequence.sites():
		if np.random.random_sample() < p_causal:
			causal_sites.append(site)
	tree_sequence_new = tree_sequence.copy(sites=causal_sites)
	m_causal = tree_sequence_new.get_num_mutations()
	return tree_sequence_new, m_causal

def case_control(y, prevalence, sample_prevalence, N):
	# Determine the liability threshold.
	p = prevalence
	T = sp.norm.ppf(1-p)
	
	# Index the cases.
	cases = [i for (i, x) in enumerate(y) if x >= T]
	mask = np.ones(len(y), dtype=bool)
	mask[cases] = False
	n_cases = len(cases)

	if sample_prevalence is None:
		n_controls = N - n_cases
	else:
		n_controls = int(((1-sample_prevalence) / sample_prevalence) * n_cases)

	controls = np.arange(N)
	if (N - n_cases) < n_controls:
		n_controls = N - n_cases
		log.log('Warning: this condition should not hold - '
			'is sample prevalence close to population prevalence?')
		controls = controls[mask]
	else:
		controls = controls[mask][random.sample(xrange(N - n_cases), n_controls)]

	controls = sorted(controls)

	return cases, controls, n_cases, n_controls, T

def run_gwas(tree_sequence, diploid_cases, diploid_controls, p_threshold, cc_maf, n_cases, n_controls):
	print 'Running GWAS'
	# Use these cases and controls to compute OR for every eligible variant.

	# Create a vector for the haplotypes of the cases - the leaves passed must be a list.
	cases = [2*x for x in diploid_cases] + [2*x+1 for x in diploid_cases]
	# Create a vector for the haplotypes of the controls - the leaves passed must be a list.
	controls = [2*x for x in diploid_controls] + [2*x+1 for x in diploid_controls]
	
	n_cases = 2*n_cases
	n_controls= 2*n_controls
	n = n_cases + n_controls
	m = tree_sequence.get_num_mutations()

	case_n_muts, control_n_muts, summary_stats = np.zeros(m), np.zeros(m), np.zeros(m)

	# Mutations are always in increasing order of position.
	k = 0
	for tree in tree_sequence.trees(tracked_leaves=cases):
		for mutation in tree.mutations():
			case_n_muts[k] = tree.get_num_tracked_leaves(mutation.node)
			k += 1

	k = 0
	for tree in tree_sequence.trees(tracked_leaves=controls):
		for mutation in tree.mutations():
			control_n_muts = tree.get_num_tracked_leaves(mutation.node)
			p = (case_n_muts[k] + control_n_muts) / n
			case_control_maf = min(p, 1-p)
			if case_control_maf > cc_maf:
				contingency = [[case_n_muts[k], n_cases - case_n_muts[k]],
							   [control_n_muts, n_controls - control_n_muts]]
				(OR, p) = sp.fisher_exact(contingency) #OR, p-value
				if not np.isnan(OR) and not np.isinf(OR) and OR != 0 and p <= p_threshold:
					summary_stats[k] = OR
				else:
					summary_stats[k] = np.nan
			else:
				summary_stats[k] = -1

			k += 1
	log.log('Done with GWAS: {ss} amenable sites.'.format(ss=np.sum(summary_stats!=-1)))
	return summary_stats

def simulate_tree_and_betas(args, log):
	h2_ldsc = np.zeros(args.n_sims, dtype=np.dtype([
		('h2_A', float), ('int_A', float),
		('h2_D', float), ('int_D', float), 
		('h2_E', float), ('int_E', float),
		('h2_AC', float), ('int_AC', float)]))
	h2_pcgc = np.zeros(args.n_sims, dtype=np.dtype([
		('h2_A', float), ('h2_D', float), ('h2_AC', float)]))
	h2_ldsc_int = np.zeros(args.n_sims, dtype=np.dtype([
		('h2_A', float), ('int_A', float),
		('h2_D', float), ('int_D', float), 
		('h2_E', float), ('int_E', float),
		('h2_AC', float), ('int_AC', float)]))

	if args.case_control:
		args.n = int(args.n_cases / args.prevalence)
		log.log('Required sample size is {N} to ensure the expected number of cases is {n}.'.format(N=args.n, n=args.n_cases))
		if args.sample_prevalence is not None and (args.prevalence > args.sample_prevalence):
			raise ValueError("Cannot set the sample prevalence lower than the population prevalence.")
		if args.sample_prevalence is not None and ((args.prevalence * 1.1) > args.sample_prevalence):
			log.log('Warning: Sample prevalence is close to population prevalence, '
				'forcing no ascertainment bias to avoid an error being thrown during sampling procedure.')
			args.sample_prevalence = None
	
	if args.free_and_no_intercept and args.no_intercept:
			raise ValueError("Can't set both no-intercept and free-and-no-intercept.")
	
	# Find out whether we need to read in recombination maps.
	if args.rec_map_chr:
		rec_map_list = []
		for chr in xrange(args.n_chr):
			rec_map_list.append(msprime.RecombinationMap.read_hapmap(sub_chr(args.rec_map_chr, chr+1)))
		args.rec, args.m = None, None
	elif args.rec_map:
		rec_map_list = [msprime.RecombinationMap.read_hapmap(args.rec_map)]*args.n_chr
		args.rec, args.m = None, None
	else:
		rec_map_list = [None]*args.n_chr

	if args.dominance is not True: args.h2_D = 0
	if args.epistasis is not True: args.h2_E = 0
	if args.gxe is not True: args.h2_AC = 0
	if args.include_pop_strat is not True: args.s2 = 0

	for sim in xrange(args.n_sims):
		if (sim == 0) or (args.fix_genetics is False):
			
			# Choose the population demographic model to use.
			if args.sim_type == 'standard':
				migration_matrix=None
				population_configurations=None
				migration_matrix=None
				demographic_events=[]
				sample_size=2*args.n
				n_pops=1

			if args.sim_type == 'out_of_africa':
				log.log("Note: passed effective population size is ignored for this option.")
				sample_size = [0, args.n, 0]
				population_configurations, migration_matrix, demographic_events, args.Ne, n_pops = mssim.out_of_africa(sample_size, args.no_migration)
				sample_size = None

			if args.sim_type == 'out_of_africa_all_pops':
				log.log("Note: passed effective population size is ignored for this option.")
				pop_sample = np.floor(args.n/3).astype(int)
				sample_size = [pop_sample, pop_sample, args.n - 2*pop_sample]
				population_configurations, migration_matrix, demographic_events, args.Ne, n_pops = mssim.out_of_africa(sample_size, args.no_migration)
				sample_size = None

			if args.sim_type == 'unicorn':
				log.log("Note: passed effective population size is ignored for this option.")
				if args.prop_EUR < 0 or args.prop_EUR > 1:
					raise ValueError("European proportion does not lie in [0,1]")
				pop_sample_eur = np.floor(args.n * args.prop_EUR)
				pop_sample_non_eur = args.n - pop_sample_eur

				non_eur_clades = int(np.floor(pop_sample_non_eur/2))
				eur_clades = int(np.floor(pop_sample_eur/3))
				ceil_eur_clades = int(np.ceil(pop_sample_eur/3))

				sample_size = [non_eur_clades, eur_clades, eur_clades, ceil_eur_clades, args.n - (non_eur_clades + 2*eur_clades + ceil_eur_clades)]
				population_configurations, migration_matrix, demographic_events, args.Ne, n_pops = mssim.unicorn(sample_size)
				sample_size = None

			log.log('Simulating genetic data using msprime:')
			
			# Create a list to fill with tree_sequences.
			tree_sequence_list = []
			tree_sequence_list_geno = []
			m_total, m_geno_total = 0, 0
			rec_map = None
			m, m_geno = np.zeros(args.n_chr).astype(int), np.zeros(args.n_chr).astype(int)
			m_start, m_geno_start = np.zeros(args.n_chr).astype(int), np.zeros(args.n_chr).astype(int)
			# If examining interaction with a covariate, pick the values of the covariate, and normalise.
			N = args.n
			C = np.random.normal(loc=0, scale=1, size=N)
			C = (C - np.mean(C)) / np.std(C)

			if args.sim_type != 'standard':
				dp = msprime.DemographyDebugger(Ne=args.Ne,
					population_configurations=population_configurations,
					migration_matrix=migration_matrix,
					demographic_events=demographic_events)
				dp.print_history()

			for chr in xrange(args.n_chr):
				tree_sequence_list.append(msprime.simulate(sample_size=sample_size,
						population_configurations=population_configurations,
						migration_matrix=migration_matrix,
						demographic_events=demographic_events,
						recombination_map=rec_map_list[chr],
						length=args.m, Ne=args.Ne,
						recombination_rate=args.rec, mutation_rate=args.mut))
				# Assign betas.
				common_mutations = []
				n_haps = tree_sequence_list[chr].get_sample_size()

				# Get the mutations > MAF.
				log.log('Determining sites > MAF cutoff {m}'.format(m=args.maf))

				for tree in tree_sequence_list[chr].trees():
					for site in tree.sites():
						f = tree.get_num_leaves(site.mutations[0].node) / n_haps
						if f > args.maf and f < 1-args.maf:
							common_mutations.append(site)

				tree_sequence_list[chr] = tree_sequence_list[chr].copy(sites=common_mutations)

				m[chr] = int(tree_sequence_list[chr].get_num_mutations())
				m_start[chr] = m_total
				m_total += m[chr]
				log.log('Number of mutations above MAF in the generated data: {m}'.format(m=m[chr]))
				log.log('Running total of sites > MAF cutoff: {m}'.format(m=m_total))

				# If genotyped proportion is < 1.
				if args.geno_prop is not None:
					tree_sequence_tmp, m_geno_tmp = set_mutations_in_tree(tree_sequence_list[chr], args.geno_prop)
					tree_sequence_list_geno.append(tree_sequence_tmp)
					m_geno[chr] = int(m_geno_tmp)
					m_geno_start[chr] = m_geno_total
					m_geno_total += m_geno[chr]
					log.log('Number of sites genotyped in the generated data: {m}'.format(m=m_geno[chr]))
					log.log('Running total of sites genotyped: {m}'.format(m=m_geno_total))
				else:
					tree_sequence_list_geno.append(tree_sequence_list[chr])
					m_geno[chr] = m[chr]
					m_geno_start[chr] = m_start[chr]
					m_geno_total = m_total

				# Do you want to write the files to .vcf?
				# The trees shouldn't be written at this point if we're 
				# using ascertained samples (which are always case control),
				# as it'll write ALL the samples to disk rather than just those that we sample.
				if args.write_trees and args.case_control is False:
					log.log('Writing genotyped information to disk')
					write_trees(args.out, tree_sequence_list_geno[chr], chr, m_geno[chr], n_pops, N, sim, args.vcf, np.arange(N))

			# Now extract the special SNP from the special chromosome.
			s = args.special_snp
			Xs = None
			if args.epistasis and s is None:
				s = int(m[args.special_chr] / 2)
			if s is not None:
				if s > m[args.special_chr]:
					raise ValueError("Index of special SNP out of bounds.")
				Xs = next(islice(tree_sequence_list_geno[args.special_chr].variants(), s, None), None)
				Xs = np.array(map(int, Xs.genotypes[0::2])) + np.array(map(int, Xs.genotypes[1::2]))

			lN_AA, lN_AD, lN_AE, lN_DA, lN_DD, lN_DE, lN_EA, lN_ED, lN_EE = (np.ones(m_geno_total) for i in range(9))

			if args.ldsc:				
				# If the optional argument to obtain LD scores from a random sub-sample of the population, set the indexing of this 
				# sub-sampling here.
				if args.ldscore_within_sample is False or args.case_control is False: # DEV: Currently we have no ascertainment for continuous traits coded up.
					if args.ldscore_sampling_prop is None:
						ldsc_index = None
						n_ldsc = N
					else:
						log.log('Using a subset of individuals from the sampled tree to determine LD scores - LD score sampling proportion: {ld}'.format(ld=args.ldscore_sampling_prop))
						n_ldsc = int(N*args.ldscore_sampling_prop)
						ldsc_index = random.sample(xrange(N), n_ldsc)

					for chr in xrange(args.n_chr):
						log.log('Determining LD scores in chromosome {chr}.'.format(chr=chr+1))
						coords = np.array(xrange(m_geno[chr]))
						block_left = getBlockLefts(coords, args.ld_wind_snps)

						start_time = time.time()
						
						lN_AA[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])] = corSumVarBlocks(tree_sequence_list_geno[chr].variants(), block_left, args.chunk_size, m_geno[chr], n_ldsc, Xs, nextSNP_add, nextSNP_add, False, ldsc_index)

						AA_time = time.time()
						log.log('Finished LD_AA scores in {T}'.format(T=pr.sec_to_str(round(AA_time-start_time, 2))))

						lN_DD[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])] = corSumVarBlocks(tree_sequence_list_geno[chr].variants(), block_left, args.chunk_size, m_geno[chr], n_ldsc, Xs, nextSNP_dom, nextSNP_dom, False, ldsc_index)

						DD_time = time.time()
						log.log('Finished LD_DD scores in {T}'.format(T=pr.sec_to_str(round(DD_time-AA_time, 2))))

						if args.epistasis:
							lN_EE[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])] = corSumVarBlocks(tree_sequence_list_geno[chr].variants(), block_left, args.chunk_size, m_geno[chr], n_ldsc, Xs, nextSNP_epi, nextSNP_epi, False, ldsc_index)
							
							EE_time = time.time()
							log.log('Finished LD_EE scores in {T}'.format(T=pr.sec_to_str(round(EE_time-DD_time, 2))))

							lN_AD[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])], lN_DA[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])] = corSumVarBlocks(tree_sequence_list_geno[chr].variants(), block_left, args.chunk_size, m_geno[chr], n_ldsc, Xs, nextSNP_add, nextSNP_dom, True, ldsc_index)
							
							AD_time = time.time() 
							log.log('Finished LD_AD, LD_DA scores in {T}'.format(T=pr.sec_to_str(round(AD_time-EE_time, 2))))

							lN_AE[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])], lN_EA[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])] = corSumVarBlocks(tree_sequence_list_geno[chr].variants(), block_left, args.chunk_size, m_geno[chr], n_ldsc, Xs, nextSNP_add, nextSNP_epi, True, ldsc_index)
							
							AE_time = time.time()
							log.log('Finished LD_AE, LD_EA scores in {T}'.format(T=pr.sec_to_str(round(AE_time-AD_time, 2))))

							lN_DE[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])], lN_ED[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])] = corSumVarBlocks(tree_sequence_list_geno[chr].variants(), block_left, args.chunk_size, m_geno[chr], n_ldsc, Xs, nextSNP_dom, nextSNP_epi, True, ldsc_index)
							
							DE_time = time.time()
							log.log('Finished LD_DE, LD_ED scores in {T}'.format(T=pr.sec_to_str(round(DE_time-AE_time, 2))))

						time_elapsed = round(time.time()-start_time,2)
						log.log('Time to evaluate LD scores: {T}'.format(T=pr.sec_to_str(time_elapsed)))

						if args.write_l2:
							if chr == 0:
								mut_names, chr_vec = np.empty(0), np.empty(0)
							mut_names = np.hstack((mut_names, np.core.defchararray.add('rs.' + str(chr+1) + ".", np.arange(1,m_geno[chr]+1).astype('str'))))
							chr_vec = np.hstack((chr_vec, np.tile(chr+1, m_geno[chr])))

					if args.write_l2:
						# Write the LD scores to disk.
						# Now, fix the chromosome number and the names of the mutations - this is as in the write_vcf function, so the 
						# resultant mutation names are identical.
						df = pd.DataFrame({'CHR':chr_vec.astype(int), 'SNP': mut_names})
						if args.epistasis:
							df['L2_AA'] = lN_AA
							df['L2_AD'] = lN_AD
							df['L2_AE'] = lN_AE
							df['L2_DA'] = lN_DA
							df['L2_DD'] = lN_DD
							df['L2_DE'] = lN_DE
							df['L2_EA'] = lN_EA
							df['L2_ED'] = lN_ED
							df['L2_EE'] = lN_EE
						else:
							df['L2_AA'] = lN_AA
							df['L2_DD'] = lN_DD
						l2_tsv = args.out + '.sim' + str(sim+1) + '.l2'
						df.to_csv(l2_tsv, sep='\t', header=True, index=False, float_format='%.3f')
						pd.set_option('display.max_rows', 200)
    					log.log('\nSummary of LD Scores in {F}'.format(F=out_fname))
    					t = df.ix[:,2:].describe()
    					log.log( t.ix[1:,:] )

		# Now, run through the chromosomes in this collection of trees, 
		# and determine the number of causal variants for each chromosome.
		start_time=time.time()
		y = np.zeros(N)

		if args.include_pop_strat is True and args.s2 > 0:
			# Get the means for the populations.
			alpha = np.random.normal(loc=0, scale=np.sqrt(args.s2), size=n_pops)
			log.log(alpha)
			# Add pop-strat additions to the phenotype vector, conditional on the population sampled from.
			for pops in xrange(n_pops):
				pop_leaves = tree_sequence_list[0].get_samples(population_id=pops)
				len(map(int, [x/2 for x in pop_leaves[0::2]]))
				y[map(int, [x/2 for x in pop_leaves[0::2]])] += alpha[pops]

		for chr in xrange(args.n_chr):			
			log.log('Picking causal variants and determining effect sizes in chromosome {chr}'.format(chr=chr+1))
			
			if (((1 + int(args.dominance) + int(args.gxe)) * args.p_causal) < 1) or args.same_causal_sites: # If the number of runs through the data is less than 1, run this speedup.
				
				tree_sequence_pheno_A, m_causal_A = set_mutations_in_tree(tree_sequence_list[chr], args.p_causal)
				log.log('Picked {m} additive causal variants out of {mc}'.format(m=m_causal_A, mc=m[chr]))

				if args.same_causal_sites is False:
					tree_sequence_pheno_D, m_causal_D = set_mutations_in_tree(tree_sequence_list[chr], args.p_causal)
					if args.h2_D > 0: log.log('Picked {m} dominance causal variants out of {mc}'.format(m=m_causal_D, mc=m[chr]))

					tree_sequence_pheno_E, m_causal_E = set_mutations_in_tree(tree_sequence_list[chr], args.p_causal)
					if args.h2_E > 0: log.log('Picked {m} epistasis causal variants out of {mc}'.format(m=m_causal_E, mc=m[chr]))

					tree_sequence_pheno_AC, m_causal_AC = set_mutations_in_tree(tree_sequence_list[chr], args.p_causal)
					if args.h2_AC > 0: log.log('Picked {m} gxe causal variants out of {mc}'.format(m=m_causal_AC, mc=m[chr]))

					if args.h2_A > 0:
						beta_A = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_total * args.p_causal)), size=m_causal_A)
						# Get the phenotypes.
						k = 0
						log.log('Determining phenotype data: additive.')
						
						for variant in progress(args.progress_bars, tree_sequence_pheno_A.variants(), total=m_causal_A): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
							X_A = nextSNP_add(variant, Xs)
							# Effect size on the phenotype.
							y += X_A * beta_A[k]
							k += 1

					if args.dominance and args.h2_D > 0:
						beta_D = np.random.normal(loc=0, scale=np.sqrt(args.h2_D / (m_total * args.p_causal)), size=m_causal_D)
						k = 0
						log.log('Determining phenotype data: dominance.')
						
						for variant in progress(args.progress_bars, tree_sequence_pheno_D.variants(), total=m_causal_D): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
							X_D = nextSNP_dom(variant, Xs)
							# Effect size on the phenotype.
							y += X_D * beta_D[k]
							k += 1

					if args.epistasis and args.h2_E > 0:
						beta_E = np.random.normal(loc=0, scale=np.sqrt(args.h2_E / (m_total * args.p_causal)), size=m_causal_E)
						k = 0
						log.log('Determining phenotype data: epistasis.')
						
						for variant in progress(args.progress_bars, tree_sequence_pheno_E.variants(), total=m_causal_E): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
							X_E = nextSNP_epi(variant, Xs)
							# Effect size on the phenotype.
							y += X_E * beta_E[k]
							k += 1

					if args.gxe and args.h2_AC > 0:
						beta_AC = np.random.normal(loc=0, scale=np.sqrt(args.h2_AC / (m_total * args.p_causal)), size=m_causal_AC)
						# If examining interaction with a covariate, pick the values of the covariate, and normalise.
						k = 0
						log.log('Determining phenotype data: gene x environment.')
						for variant in progress(args.progress_bars, tree_sequence_pheno_AC.variants(), total=m_causal_AC): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
							X_A = nextSNP_add(variant)
							# Effect size on the phenotype.
							y += C * X_A * beta_AC[k]
							k += 1
				else:
					beta_A, beta_D, beta_E, beta_AC = np.zeros(m_causal_A), np.zeros(m_causal_A), np.zeros(m_causal_A), np.zeros(m_causal_A)  
					if args.h2_A > 0:
						beta_A = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_total * args.p_causal)), size=m_causal_A)
					
					if args.dominance and args.h2_D > 0:
						beta_D = np.random.normal(loc=0, scale=np.sqrt(args.h2_D / (m_total * args.p_causal)), size=m_causal_A)

					if args.epistasis and args.h2_E > 0:
						beta_E = np.random.normal(loc=0, scale=np.sqrt(args.h2_E / (m_total * args.p_causal)), size=m_causal_A)

					if args.gxe and args.h2_AC > 0:
						beta_AC = np.random.normal(loc=0, scale=np.sqrt(args.h2_AC / (m_total * args.p_causal)), size=m_causal_A)

					k = 0
					log.log('Determining phenotype data')
					# Note that we use just one tree_sequence here, because the causal sites are the same in this portion of the code.
					for variant in progress(args.progress_bars, tree_sequence_pheno_A.variants(), total=m_causal_A): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
						X_A, X_D, X_E = nextSNP(variant, Xs)
						# Effect size on the phenotype.
						y += X_A * beta_A[k] + X_D * beta_D[k] + X_E * beta_E[k] + C * X_A * beta_AC[k]
						k += 1

			else:
				m_causal = int(m[chr] * args.p_causal)
				beta_A, beta_D, beta_E, beta_AC = [np.zeros(m[chr]) for i in range(4)]
				beta_A_causal_index = random.sample(xrange(m[chr]), m_causal)
				log.log('Picked {m} additive causal variants out of {mc}'.format(m=m_causal, mc=m[chr]))

				if args.h2_A > 0:
					beta_A[beta_A_causal_index] = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_total * args.p_causal)), size=m_causal)
				
				if args.dominance:
					beta_D, beta_D_causal_index = np.zeros(m[chr]), random.sample(xrange(m[chr]), m_causal)
					log.log('Picked {m} dominance causal variants out of {mc}'.format(m=m_causal, mc=m[chr]))
					if args.h2_D > 0:
						beta_D[beta_D_causal_index] = np.random.normal(loc=0, scale=np.sqrt(args.h2_D / (m_total * args.p_causal)), size=m_causal)

				if args.epistasis:
					beta_E, beta_E_causal_index = np.zeros(m[chr]), random.sample(xrange(m[chr]), m_causal)
					log.log('Picked {m} epistasis causal variants out of {mc}'.format(m=m_causal, mc=m[chr]))
					if args.h2_E > 0:
						beta_E[beta_E_causal_index] = np.random.normal(loc=0, scale=np.sqrt(args.h2_E / (m_total * args.p_causal)), size=m_causal)

				if args.gxe:
					beta_AC, beta_AC_causal_index = np.zeros(m[chr]), random.sample(xrange(m[chr]), mc=m_causal)
					log.log('Picked {m} gxe causal variants out of {mc}'.format(m=m_causal, mc=m[chr]))
					if args.h2_AC > 0:
						beta_AC[beta_AC_causal_index] = np.random.normal(loc=0, scale=np.sqrt(args.h2_AC / (m_total * args.p_causal)), size=m_causal)

				# Get the phenotypes.
				k = 0
				log.log('Determining phenotype data.')

				for variant in progress(args.progress_bars, tree_sequence_list[chr].variants(), total=m[chr]): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
					X_A, X_D, X_E = nextSNP(variant, Xs)
					# Effect size on the phenotype.
					y += X_A * beta_A[k] + X_D * beta_D[k] + X_E * beta_E[k] + X_A * C * beta_AC[k]
					k += 1
		
		# Add noise to the y.
		y += np.random.normal(loc=0, scale=np.sqrt(1-(args.h2_A+args.h2_D+args.h2_E+args.h2_AC+args.s2)), size=N)
		# Finally, normalise.
		y = (y - np.mean(y)) / np.std(y)
		time_elapsed = round(time.time()-start_time,2)
		log.log('Time to evaluate phenotype data: {T}'.format(T=pr.sec_to_str(time_elapsed)))

		# Initialise the chi squared statistics.
		chisq_A, chisq_D, chisq_E, chisq_AC = np.zeros((m_geno_total,1)), np.zeros((m_geno_total,1)), np.zeros((m_geno_total,1)), np.zeros((m_geno_total,1))

		if args.case_control:
			log.log("Running case-control simulation.")
			if args.prevalence is None:
				raise ValueError("prevalence must be set if running case-control analysis.")
			cases, controls, n_cases, n_controls, T = case_control(y, args.prevalence, args.sample_prevalence, N)
			n = n_cases + n_controls
			y_cc = np.zeros(n)
			y_cc[:n_cases] = 1
			index = cases + controls
			C_sim = C[index]

			if args.linear is False and args.ldsc is True:
				k = 0
				for chr in xrange(args.n_chr):
					for variant in progress(args.progress_bars, tree_sequence_list_geno[chr].variants(), total=m[chr]):
						X_A, X_D, X_E = nextSNP(variant, Xs, index = index)
						chisq_A[k] = sm.Logit(y_cc, sm.add_constant(X_A)).fit(disp=0).llr
						chisq_D[k] = sm.Logit(y_cc, sm.add_constant(X_D)).fit(disp=0).llr
						chisq_E[k] = sm.Logit(y_cc, sm.add_constant(X_E)).fit(disp=0).llr
						chisq_AC[k] = sm.Logit(y_cc, sm.add_constant(C_sim * X_A)).fit(disp=0).llr
						k += 1

		if (((args.case_control is False) or (args.case_control is True and args.linear is True)) and args.ldsc is True):
			if args.case_control:
				log.log("Warning: running linear regression for case-control.")
				y = (y_cc - np.mean(y_cc)) / np.std(y_cc)
				index = cases + controls
				C_sim = C[index]
			else:
				index = None
				C_sim = C
				n = N

			# Then use these ys to determine beta hats.
			k = 0
			for chr in xrange(args.n_chr):
				log.log('Determining chi-squared statistics in chromosome {chr}'.format(chr=chr+1))
				for variant in tree_sequence_list_geno[chr].variants():
					X_A, X_D, X_E = nextSNP(variant, Xs, index=index)
					# Then sum to get the effect size on the phenotype.
					chisq_A[k] = np.dot(y.reshape(1,n), X_A)**2 / n
					chisq_D[k] = np.dot(y.reshape(1,n), X_D)**2 / n
					chisq_E[k] = np.dot(y.reshape(1,n), X_E)**2 / n
					chisq_AC[k] = np.dot(y.reshape(1,n), C_sim * X_A)**2 / n
					k += 1

		if args.write_pheno:
			if args.case_control:
				sample_ID = index
				y = y_cc.astype(int)
				# print sample_ID

				if args.write_trees:
					tree_index = [[2*x,2*x+1] for x in index]
					tree_index = [j for x in tree_index for j in x]

					for chr in xrange(args.n_chr):
						tree_sequence_to_write = tree_sequence_list_geno[chr].simplify(tree_index)
						write_trees(args.out, tree_sequence_to_write, chr, m_geno[chr], n_pops, N, sim, args.vcf, index)

			else:
				sample_ID = np.arange(N)
			df_pheno=pd.DataFrame({'sample_ID':sample_ID, 'phenotype':y})
			df_pheno.to_csv(args.out + '.sim' + str(sim+1) + '.pheno.tsv', sep='\t', header=True, index=False)


		# If the optional argument to obtain LD scores from a random sub-sample of the population, set the indexing of this 
		# sub-sampling here.
		if args.ldsc:
			if args.ldscore_within_sample is True and args.case_control is True: # DEV: Currently we have no ascertainment for continuous traits coded up.
				if args.ldscore_sampling_prop is None:
					ldsc_index = index
					n_ldsc = n
				else:
					log.log('Using a subset of individuals from the sampled tree to determine LD scores - LD score sampling proportion: {ld}'.format(ld=args.ldscore_sampling_prop))
					n_ldsc = int(n*args.ldscore_sampling_prop)
					ldsc_index = random.sample(index, n_ldsc)

				for chr in xrange(args.n_chr):
					log.log('Determining LD scores in chromosome {chr}.'.format(chr=chr+1))
					coords = np.array(xrange(m_geno[chr]))
					block_left = getBlockLefts(coords, args.ld_wind_snps)
					
					start_time=time.time()
					lN_A[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])], lN_D[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])] = corSumVarBlocks(tree_sequence_list_geno[chr].variants(), block_left, args.chunk_size, m_geno[chr], n_ldsc, ldsc_index)
					time_elapsed = round(time.time()-start_time,2)
					log.log('Time to evaluate LD scores in chromosome {chr}: {T}'.format(T=pr.sec_to_str(time_elapsed)))

			# Intercept options for the regression.
			intercept_h2 = [None]
			if args.free_and_no_intercept: intercept_h2 = [None, 1]
			if args.no_intercept: intercept_h2 = [1]

			# Run the regressions
			log.log('Running LD score regressions.')
			hsqhat_A, hsqhat_D, hsqhat_E, hsqhat_AC = [], [], [], []
			for i in xrange(len(intercept_h2)):
				if args.epistasis:
					y = np.concatenate((chisq_A, chisq_D, chisq_E), axis=0)
					x = np.concatenate((lN_AA, lN_DA, lN_EA, lN_AD, lN_DD, lN_ED, lN_AE, lN_DE, lN_EE))[None,:].reshape((3,3*m_geno_total)).T
					w = np.concatenate((lN_EA, lN_ED, lN_EE))[None,:].T

					hsqhat_E.append(reg.Hsq(y, x, w,
					np.tile(n,3*m_geno_total).reshape((3*m_geno_total,1)), np.tile(m_geno_total,3).reshape((1,3)),
					n_blocks=min(m_geno_total, args.n_blocks), intercept=intercept_h2[i]))

				else:
					hsqhat_A.append(reg.Hsq(chisq_A,
					lN_AA.reshape((m_geno_total,1)), lN_AA.reshape((m_geno_total,1)),
					np.tile(n,m_geno_total).reshape((m_geno_total,1)), np.array(m_geno_total).reshape((1,1)),
					n_blocks=min(m_geno_total, args.n_blocks), intercept=intercept_h2[i]))

					hsqhat_D.append(reg.Hsq(chisq_D,
					lN_DD.reshape((m_geno_total,1)), lN_DD.reshape((m_geno_total,1)),
					np.tile(n,m_geno_total).reshape((m_geno_total,1)), np.array(m_geno_total).reshape((1,1)),
					n_blocks=min(m_geno_total, args.n_blocks), intercept=intercept_h2[i]))

				hsqhat_AC.append(reg.Hsq(chisq_AC,
				lN_AA.reshape((m_geno_total,1)), lN_AA.reshape((m_geno_total,1)),
				np.tile(n,m_geno_total).reshape((m_geno_total,1)), np.array(m_geno_total).reshape((1,1)),
				n_blocks=min(m_geno_total, args.n_blocks), intercept=intercept_h2[i]))

		if args.case_control:
			study_prevalence = n_cases / n
			scaling = (args.prevalence**2 * (1-args.prevalence)**2) / ((study_prevalence * (1 - study_prevalence)) * sp.norm.pdf(T)**2)
		else:
			scaling = 1

		if args.pcgc:
			if args.case_control:
				# Make sure that the phenotype is standardised.
				y = (y_cc - np.mean(y_cc)) / np.std(y_cc)

			P = np.outer(y, y)

			if (sim == 0) or (args.fix_genetics is False) or (args.case_control):
				where = np.triu_indices(n, k=1)
				K_A, K_D, K_E, K_AC = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
				for chr in xrange(args.n_chr):
					log.log('Determining K_A, K_D, and K_E in chromosome {chr}'.format(chr=chr+1))
					K_A, K_D, K_E, K_AC = obtain_K(tree_sequence_list_geno[chr].variants(),
						K_A, K_D, K_E, K_AC, C_sim, args.chunk_size, m[chr], n,
						args.progress_bars, index)
					log.log('Time to evaluate K_A, K_D, and K_E in chromosome {chr}: {T}'.format(chr=chr+1, T=pr.sec_to_str(time_elapsed)))
					
			log.log('Running PCGC regressions')
			h2_A = sm.OLS(P[where], exog = K_A[where] / m_geno_total).fit().params
			h2_AD = sm.OLS(P[where], exog = np.column_stack((K_A[where], K_D[where])) / m_geno_total).fit().params
			h2_ADAC = sm.OLS(P[where], exog = np.column_stack((K_A[where], K_D[where], K_AC[where])) / m_geno_total).fit().params

			h2_pcgc['h2_A'][sim] = (h2_A) * scaling
			h2_pcgc['h2_D'][sim] = (np.sum(h2_AD) - h2_A) * scaling
			h2_pcgc['h2_AC'][sim] = (np.sum(h2_ADAC) - np.sum(h2_AD)) * scaling

		if args.ldsc is True:
			if args.epistasis:
				h2_ldsc['h2_A'][sim], h2_ldsc['h2_D'][sim], h2_ldsc['h2_E'][sim], h2_ldsc['h2_AC'][sim] = np.array([hsqhat_E[0].cat[0,0], hsqhat_E[0].cat[0,1], hsqhat_E[0].cat[0,2], hsqhat_AC[0].tot]) * scaling
				h2_ldsc['int_A'][sim], h2_ldsc['int_D'][sim], h2_ldsc['int_E'][sim], h2_ldsc['int_AC'][sim] = hsqhat_E[0].intercept, hsqhat_E[0].intercept, hsqhat_E[0].intercept, hsqhat_AC[0].intercept
			else:
				h2_ldsc['h2_A'][sim], h2_ldsc['h2_D'][sim], h2_ldsc['h2_AC'][sim] = np.array([hsqhat_A[0].tot, hsqhat_D[0].tot, hsqhat_AC[0].tot]) * scaling
				h2_ldsc['int_A'][sim], h2_ldsc['int_D'][sim], h2_ldsc['int_AC'][sim] = hsqhat_A[0].intercept, hsqhat_D[0].intercept, hsqhat_AC[0].intercept

			if args.free_and_no_intercept:
				h2_ldsc_int['h2_A'][sim], h2_ldsc_int['h2_D'][sim], h2_ldsc_int['h2_AC'][sim] = np.array([hsqhat_A[1].tot, hsqhat_D[1].tot, hsqhat_AC[1].tot]) * scaling
				h2_ldsc_int['int_A'][sim], h2_ldsc_int['int_D'][sim], h2_ldsc_int['int_AC'][sim] = hsqhat_A[1].intercept, hsqhat_D[1].intercept, hsqhat_AC[1].intercept

	return h2_ldsc, h2_pcgc, h2_ldsc_int

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='msprimesim', type=str,
	help='Output filename prefix. This will be an output of heritability estimates'
	' across the n_sims.')
parser.add_argument('--h2_A', default=0.3, type=float,
	help='Additive heritability contribution [Default: 0.3].')
parser.add_argument('--h2_D', default=0.1, type=float,
	help='Dominance heritability contribution [Default: 0.1].')
parser.add_argument('--h2_E', default=0, type=float,
	help='Epistasis heritability contribution [Default: 0].')
parser.add_argument('--h2_AC', default=0.2, type=float,
	help='Dominance heritability contribution [Default: 0.2].')
parser.add_argument('--p-causal', default=1, type=float,
	help='Proportion of SNPs that are causal [Default: 1].')
parser.add_argument('--special-snp', default=None, type=float,
	help='Index of special SNP for epistasis.')
parser.add_argument('--special-chr', default=0, type=float,
	help='Index of special chromosome for epistasis.')
# Filtering / Data Management for LD Score
parser.add_argument('--ld-wind-snps', default=1000, type=int,
	help='Specify the window size to be used for estimating LD Scores in units of '
	'# of SNPs [Default: 1000].')
# Flags you should almost never use
parser.add_argument('--chunk-size', default=50, type=int,
	help='Chunk size for LD Score calculation. Use the default [Default: 50].')
parser.add_argument('--n', default=40000, type=int,
	help='Number of individuals in the sampled genotypes [Default: 40,000].')
parser.add_argument('--m', default=1000000, type=int,
	help='Length of the region analysed in nucleotides [Default: 1,000,000].')
parser.add_argument('--Ne', default=10000, type=int,
	help='Effective population size [Default: 10,000].')
parser.add_argument('--sim-type', default='standard', type=str,
	help='Type of simulation to run. Currently recognises "standard",'
	'"out-of-africa" and "out-of-africa-all-pops" [Default: standard]')
parser.add_argument('--maf', default=0.05, type=float,
	help='The minor allele frequency cut-off [Default: 0.05].')
parser.add_argument('--rec', default=2e-8, type=float,
	help='Recombination rate across the region [Default: 2e-8].')
parser.add_argument('--mut', default=2e-8, type=float,
	help='Mutation rate across the region [Default: 2e-8].')
parser.add_argument('--no-intercept', action='store_true',
	help = 'This constrains the LD Score regression intercept to equal 1.')
parser.add_argument('--free-and-no-intercept', action='store_true',
	help = 'This runs both a free and a constrained intercept equal to 1.')
parser.add_argument('--n-sims', default=1, type=int,
	help='Number of msprime simulations to run [Default: 1].')
parser.add_argument('--n-blocks', default=200, type=int,
	help='Number of block jackknife blocks [Default: 200].')
parser.add_argument('--fix-genetics', default=False, action='store_true',
	help='Fix the genetic data to be the same across runs.')
parser.add_argument('--n-chr', default=1, type=int,
	help='Number of chromosomes to simulate if --sim-chr is set [Default: 1].')
parser.add_argument('--rec-map', default=None, type=str,
	help='If you want to pass a recombination map, include the filepath here.')
parser.add_argument('--rec-map-chr', default=None, type=str,
	help='If you want to pass a recombination map, include the filepath here. '
	'The filename should contain the symbol @, msprimesim will replace instances '
	'of @ with chromosome numbers.')
parser.add_argument('--pcgc', default=False, action='store_true',
	help='Do you want to estimate heritability using PCGC too? Warning: slow and memory intensive.')
parser.add_argument('--dominance', default=False, action='store_true',
	help='Do you want to include dominance simulation and estimation?')
parser.add_argument('--epistasis', default=False, action='store_true',
	help='Do you want to include epistasis simulation and estimation?')
parser.add_argument('--gxe', default=False, action='store_true',
	help='Do you want to include a gene by environment interaction simulation and estimation?')
parser.add_argument('--write-trees', default=False, action='store_true',
	help='Do you want to write the tree sequences to .bim/.bed/.fam format?')
parser.add_argument('--geno-prop', default=None, type=float,
	help='Is the proportion of SNPs genotyped different to the number of SNPs?')
parser.add_argument('--progress-bars', default=False, action='store_true',
	help='Do you want fancy progress bars? Important - Don\'t use this flag when running jobs on a '
	'cluster, lest you want tons of printing to your .log file!')
parser.add_argument('--same-causal-sites', default=False, action='store_true', 
	help='Are the causal sites the same across additive, dominance, and gene x environment components?')
parser.add_argument('--case-control', default=False, action='store_true', 
	help='Do you want to run a case-control GWAS?')
parser.add_argument('--prevalence', default=0.1, type=float,
	help='If running a case-control GWAS, what is the prevalence of the disorder? [Default: 0.1]')
parser.add_argument('--linear', default=False, action='store_true',
	help='Do you want to run linear regression when using the case-control flag?')
parser.add_argument('--n-cases', default=1000, type=float,
	help='If running a case-control study simulation, the expected number of cases given the provided prevalence '
	'in the population [Default: 1000].')
parser.add_argument('--sample-prevalence', default=None, type=float,
	help='If running a case-control study simulation, the prevalence of the cases in the study sample [Default: None]. '
	'The \'None\' default keeps the study prevalence the same as the population prevalence.')
parser.add_argument('--ldscore-sampling-prop', default=None, type=float,
	help='If running a large ascertained case-control simulation, you may want to determine LD score estimates a '
	'subset of the individuals in the tree to increase speed.')
parser.add_argument('--ldscore-within-sample', default=False, action='store_true',
	help='Do you want to evaluate the LD scores using the case-control sample used to obtain the effect size estimates? '
	'Note that this will result in LD scores being generated for each simulation, so may slow things down if ascertainment is '
	'low and the sample size is large.')
parser.add_argument('--ldsc', default=False, action='store_true',
	help='Do we perform LD score regression?')
parser.add_argument('--include-pop-strat', default=False, action='store_true',
	help='Do we include population stratification in the contribution to the phenotype. As default, we randomly draw the mean from '
	'a Normal with mean 0 and variance 1')
parser.add_argument('--s2', default=0.1, type=float,
	help='What is the clade associated variance in the phenotype?')
parser.add_argument('--no-migration', default=False, action='store_true',
	help='Turn off migration in the demographic history - currently only has an effect for \'out_of_africa\' in the sim-type flag')
parser.add_argument('--vcf', default=False, action='store_true', 
	help='If saving the trees, do we want to save as .vcf file? Default is PLINK .bed/.bim/.fam format if this flag is not used.')
parser.add_argument('--write-pheno', default=False, action='store_true', 
	help='Do you want to write the phenotypes to disk?')
parser.add_argument('--prop-EUR', default=0.75, type=float,
	help='What proportion of samples does the European portion make up?')
parser.add_argument('--write-l2', default=False, action='store_true', 
	help='Do you want to write the LD scores for this simulation to disk?')


if __name__ == '__main__':

	args = parser.parse_args()
	log = pr.Logger(args.out+'.msprime.log')
	try:
		defaults = vars(parser.parse_args(''))
		opts = vars(args)
		non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
		header = ''
		header += 'Call: \n'
		header += './msprimesim.py \\\n'
		options = ['--'+x.replace('_','-')+' '+str(opts[x])+' \\' for x in non_defaults]
		header += '\n'.join(options).replace('True','').replace('False','')
		header = header[0:-1]+'\n'
		log.log(header)
		log.log('Beginning analysis at {T}'.format(T=time.ctime()))
		start_time = time.time()
		out_fname = args.out + '.h2'

		if args.pcgc:
			h2_ldsc, h2_pcgc, h2_ldsc_int = simulate_tree_and_betas(args, log)
			out_fname_pcgc = args.out + '.pcgc'
			
			df = pd.DataFrame.from_records(np.c_[h2_pcgc['h2_A'], h2_pcgc['h2_D'], h2_pcgc['h2_AC']])
			df.columns = ['h2_A', 'h2_D', 'h2_AC']
			df.to_csv(out_fname_pcgc, sep='\t', header=True, index=False, float_format='%.3f')
		else:
			h2_ldsc, h2_pcgc, h2_ldsc_int = simulate_tree_and_betas(args, log)

		df = pd.DataFrame.from_records(np.c_[
			h2_ldsc['h2_A'], h2_ldsc['int_A'],
			h2_ldsc['h2_D'], h2_ldsc['int_D'],
			h2_ldsc['h2_E'], h2_ldsc['int_E'],
			h2_ldsc['h2_AC'], h2_ldsc['int_AC']])
		df.columns = ['h2_A', 'int_A', 'h2_D', 'int_D', 'h2_E', 'int_E', 'h2_AC', 'int_AC']
		df.to_csv(out_fname, sep='\t', header=True, index=False, float_format='%.3f')

		if args.free_and_no_intercept:
			out_fname = args.out + '.int.h2'

			df = pd.DataFrame.from_records(np.c_[
			h2_ldsc_int['h2_A'], h2_ldsc_int['int_A'],
			h2_ldsc_int['h2_D'], h2_ldsc_int['int_D'],
			h2_ldsc_int['h2_AC'], h2_ldsc_int['int_AC']])
			df.columns = ['h2_A', 'int_A', 'h2_D', 'int_D', 'h2_AC', 'int_AC']
			df.to_csv(out_fname, sep='\t', header=True, index=False, float_format='%.3f')

	except Exception:
		ex_type, ex, tb = sys.exc_info()
		log.log( traceback.format_exc(ex) )
		raise
	finally:
		log.log('Analysis finished at {T}'.format(T=time.ctime()) )
		time_elapsed = round(time.time()-start_time,2)
		log.log('Total time elapsed: {T}'.format(T=pr.sec_to_str(time_elapsed)))

