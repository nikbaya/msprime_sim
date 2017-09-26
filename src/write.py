from __future__ import division
import msprime
import pandas as pd
import numpy as np
import os

def trees(out, tree_sequence, chr, m, n_pops, N, sim, vcf, sample_index):
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