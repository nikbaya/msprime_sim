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
    for i in xrange(M):
        while j < M and abs(coords[j] - coords[i]) > max_dist:
            j += 1
        block_left[i] = j
    return block_left

def corSumVarBlocks(variants, block_left, c, m, N, Xs, snp_getter1, snp_getter2, both, ldsc_index):
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

def get_ldscores(args, m_geno, m_geno_start, m_geno_total, tree_sequence_list_geno, n_ldsc, ldsc_index, sim, log):
	s = args.special_snp
	Xs = None
	if args.epistasis and s is None:
		s = int(m_geno[args.special_chr] / 2)
	if s is not None:
		if s > m_geno[args.special_chr]:
			raise ValueError('Index of special SNP out of bounds.')
		Xs = next(islice(tree_sequence_list_geno[args.special_chr].variants(), s, None), None)
		Xs = np.array(map(int, Xs.genotypes[0::2])) + np.array(map(int, Xs.genotypes[1::2]))

	lN_AA, lN_AD, lN_AE, lN_DA, lN_DD, lN_DE, lN_EA, lN_ED, lN_EE = (np.ones(m_geno_total) for i in range(9))

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
		df = pd.DataFrame({'CHR':chr_vec.astype(int), 'SNP': mut_names})
		if args.epistasis:
			df = df.join(pd.DataFrame({'L2_AA':lN_AA, 'L2_AD':lN_AD, 'L2_AE':lN_AE, 'L2_DA':lN_DA, 'L2_DD':lN_DD, 'L2_DE':lN_DE, 'L2_EA':lN_EA, 'L2_ED':lN_ED, 'L2_EE':lN_EE}))
		else:
			df = df.join(pd.DataFrame({'L2_AA':lN_AA, 'L2_DD':lN_DD}))
			
		l2_tsv = args.out + '.sim' + str(sim+1) + '.l2'
		df.to_csv(l2_tsv, sep='\t', header=True, index=False, float_format='%.3f')
		pd.set_option('display.max_rows', 200)
		log.log('\nSummary of LD Scores in {F}'.format(F=out_fname))
		t = df.ix[:,2:].describe()
		log.log( t.ix[1:,:] )

	if args.epistasis:
		return lN_AA, lN_AD, lN_AE, lN_DA, lN_DD, lN_DE, lN_EA, lN_ED, lN_EE
	else:
		return lN_AA, lN_DD
