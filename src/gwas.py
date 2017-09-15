from __future__ import division
import msprime
import numpy as np
import scipy.stats as sp

def run_gwas(tree_sequence, diploid_cases, diploid_controls, p_threshold, cc_maf, n_cases, n_controls, log):
	log.log('Running GWAS')
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
