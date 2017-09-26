from __future__ import division
import msprime
import numpy as np
import random
import tqdm
import scipy.stats as sp
import src.regressions as reg
import src.tools as tl
import src.snpgetter as sg
import src.printing as pr
import time, sys, traceback, argparse
import statsmodels.api as sm

def obtain_K(variants, K_A, K_D, K_AC, C, c, m, n, progress_bars, index):

	# Number of chunks of length c in m sites.
	n_c = np.floor(m/c).astype(int)
	X_A, X_D = np.empty((n, c)), np.empty((n, c))
	C_mat = np.repeat(C, c).reshape((n, c))

	for i in tl.progress(progress_bars, xrange(n_c), total=n_c):
		k = 0

		while k < c:
			variant = variants.next()
			X_A[:,k], X_D[:,k] = sg.nextSNP(variant, index)
			k += 1
		
		K_A += np.dot(X_A, X_A.T)
		K_D += np.dot(X_D, X_D.T)
		K_AC += np.dot(C_mat * X_A, (C_mat * X_A).T)
		
	# The final chunk.
	if (n_c * c) < m:
		k = 0
		c = m - (n_c * c)
		X_A, X_D = np.empty((n, c)), np.empty((n, c))
		C_mat = np.repeat(C, c).reshape((n, c))

		while k < c:
			variant = variants.next()
			X_A[:,k], X_D[:,k] = sg.nextSNP(variant, index)
			k += 1

		K_A += np.dot(X_A, X_A.T)
		K_D += np.dot(X_D, X_D.T)
		K_AC += np.dot(C_mat * X_A, (C_mat * X_A).T)

	return K_A, K_D, K_AC

def pcgc(args, sim, tree_sequence_list_geno, y, h2_pcgc, n, C_sim, index, m_geno_total, scaling, log):

	P = np.outer(y, y)
	if (sim == 0) or (args.fix_genetics is False) or (args.case_control):
		where = np.triu_indices(n, k=1)
		K_A, K_D, K_AC = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
		for chr in xrange(args.n_chr):
			m_geno_chr = tree_sequence_list_geno[chr].get_num_mutations()
			log.log('Determining K_A and K_D in chromosome {chr}'.format(chr=chr+1))
			start_time = time.time()
			K_A, K_D, K_AC = obtain_K(tree_sequence_list_geno[chr].variants(),
				K_A, K_D, K_AC, C_sim, args.chunk_size, m_geno_chr, n,
				args.progress_bars, index)
			time_elapsed = round(time.time()-start_time,2)
			log.log('Time to evaluate K_A and K_D in chromosome {chr}: {T}'.format(chr=chr+1, T=pr.sec_to_str(time_elapsed)))
			
	log.log('Running PCGC regressions')
	h2_A = sm.OLS(P[where], exog = K_A[where] / m_geno_total).fit().params
	h2_AD = sm.OLS(P[where], exog = np.column_stack((K_A[where], K_D[where])) / m_geno_total).fit().params
	h2_ADAC = sm.OLS(P[where], exog = np.column_stack((K_A[where], K_D[where], K_AC[where])) / m_geno_total).fit().params

	h2_pcgc['h2_A'][sim] = (h2_A) * scaling
	h2_pcgc['h2_D'][sim] = (np.sum(h2_AD) - h2_A) * scaling
	h2_pcgc['h2_AC'][sim] = (np.sum(h2_ADAC) - np.sum(h2_AD)) * scaling

	return h2_pcgc
