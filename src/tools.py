from __future__ import division
import numpy as np
import scipy.stats as sp
import msprime

def initial_warnings_and_parsing(args, log):
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
	
	if args.load_tree_sequence is not None:
		args.fix_genetics = True

	# Find out whether we need to read in recombination maps.
	if args.rec_map_chr:
		rec_map_list = []
		for chr in xrange(args.n_chr):
			rec_map_list.append(msprime.RecombinationMap.read_hapmap(tl.sub_chr(args.rec_map_chr, chr+1)))
		args.rec, args.m = None, None
	elif args.rec_map:
		rec_map_list = [msprime.RecombinationMap.read_hapmap(args.rec_map)]*args.n_chr
		args.rec, args.m = None, None
	else:
		rec_map_list = [None]*args.n_chr

	if args.dominance is not True: args.h2_D = 0
	if args.gxe is not True: args.h2_AC = 0
	if args.include_pop_strat is not True: args.s2 = 0

	return args, rec_map_list

def p_to_z(p, N):
    # Convert p-value and N to standardized beta.
    return np.sqrt(sp.chi2.isf(p, 1))

def sub_chr(s, chr):
    # Substitute chr for @, else append chr to the end of str.
    if '@' not in s:
        s += '@'
    return s.replace('@', str(chr))

def progress(progress_bars, range_object, total):
	if progress_bars:
		return tqdm.tqdm(range_object, total=total)
	else:
		return range_object
