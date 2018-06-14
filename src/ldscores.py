from __future__ import division
import numpy as np
import src.snpgetter as sg
import time, sys, traceback, argparse
import src.printing as pr
import pandas as pd

def l2(x, n):
	sq = np.square(x)
	return sq - 1/n

def getBlockLefts(coords, max_dist):
    M = len(coords)
    j = 0
    block_left = np.zeros(M)
    for i in range(M):
        while j < M and abs(coords[j] - coords[i]) > max_dist:
            j += 1
        block_left[i] = j
    return block_left

def corSumVarBlocks(variants, block_left, c, m, N, ldsc_index):

	block_sizes = np.array(np.arange(m) - block_left)
	block_sizes = (np.ceil(block_sizes / c) * c).astype(int)
	cor_sum_A, cor_sum_D = np.zeros(m), np.zeros(m)

	# b = index of first SNP for which SNP 0 is not included in LD Score
	b = np.nonzero(block_left > 0)
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
	
	A = np.empty((N,b))
	while k < b:
		variant = variants.__next__()
		A[:,k] = sg.nextSNP_add(variant, index=ldsc_index)
		k += 1

	rfuncAB = np.zeros((b, c))
	rfuncBB = np.zeros((c, c))
	# Chunk inside of block
	for l_B in range(0, b, c):  # l_B := index of leftmost SNP in matrix B
		B = A[:, l_B:l_B+c]
		np.dot(A.T, B / N, out=rfuncAB)
		rfuncAB = l2(rfuncAB, N)
		cor_sum_A[l_A:l_A+b] += np.sum(rfuncAB, axis=1)
		cor_sum_D[l_A:l_A+b] += np.sum(np.copy(rfuncAB)**2, axis=1)

	# Chunk to right of block
	b0 = int(b)
	md = int(c*np.floor(m/c))
	end = md + 1 if md != m else md

	for l_B in range(b0, end, c):
		# Update the block
		old_b = b

		b = block_sizes[l_B]
		if l_B > b0 and b > 0:
			# block_size can't increase more than c
			# block_size can't be less than c unless it is zero
			# Both of these things make sense
			A = np.hstack((A[:, old_b-b+c:old_b], B))
			l_A += old_b-b+c
		elif l_B == b0 and b > 0:
			A = A[:, b0-b:b0]
			l_A = b0-b
		elif b == 0:  # no SNPs to left in window, e.g., after a sequence gap
			A = np.array(()).reshape((N, 0))
			l_A = l_B
		if l_B == md:
			c = m - md
			rfuncAB = np.zeros((b, c))
			rfuncBB = np.zeros((c, c))
		if b != old_b:
			rfuncAB = np.zeros((b, c))

		k = 0
		B = np.empty((N,c))
		while k < c:
			variant = variants.__next__()
			B[:,k] = sg.nextSNP_add(variant, index=ldsc_index)
			k += 1

		np.dot(A.T, B / N, out=rfuncAB)
		rfuncAB = l2(rfuncAB, N)

		cor_sum_A[l_A:l_A+b] += np.sum(rfuncAB, axis=1)
		cor_sum_A[l_B:l_B+c] += np.sum(rfuncAB, axis=0)

		rfuncAB2 = np.copy(rfuncAB)**2
		cor_sum_D[l_A:l_A+b] += np.sum(rfuncAB2, axis=1)
		cor_sum_D[l_B:l_B+c] += np.sum(rfuncAB2, axis=0)

		np.dot(B.T, B / N, out=rfuncBB)
		rfuncBB = l2(rfuncBB, N)
		cor_sum_A[l_B:l_B+c] += np.sum(rfuncBB, axis=0)
		cor_sum_D[l_B:l_B+c] += np.sum(np.copy(rfuncBB)**2, axis=0)

	return cor_sum_A, cor_sum_D

def get_ldscores(args, m_geno, m_geno_start, m_geno_total, tree_sequence_list_geno, n_ldsc, ldsc_index, sim, log):

	lN_A, lN_D = np.ones(m_geno_total), np.ones(m_geno_total)

	for chr in range(args.n_chr):
		log.log('Determining LD scores in chromosome {chr}.'.format(chr=chr+1))
		coords = np.array(range(m_geno[chr]))
		block_left = getBlockLefts(coords, args.ld_wind_snps)
		
		start_time=time.time()
		lN_A[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])], lN_D[m_geno_start[chr]:(m_geno_start[chr]+m_geno[chr])] = corSumVarBlocks(tree_sequence_list_geno[chr].variants(), block_left, args.chunk_size, m_geno[chr], n_ldsc, ldsc_index)
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
		d={'CHR':chr_vec.astype(int), 'SNP': mut_names, 'L2_AA':lN_A, 'L2_DD':lN_D}
		df=pd.DataFrame(d)
		df = df[['CHR', 'SNP', 'L2_AA', 'L2_DD']]
		l2_tsv = args.out + ".sim." + str(sim+1) + '.l2'
		df.to_csv(l2_tsv, sep='\t', header=True, index=False, float_format='%.3f')

	return lN_A, lN_D
