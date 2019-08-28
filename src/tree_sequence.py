from __future__ import division
import msprime
import numpy as np
import src.msprime_sim_scenarios as mssim
import src.write as write

def initialise(args):
	tree_sequence_list = []
	tree_sequence_list_geno = []
	genotyped_list_index = []
	m_total, m_geno_total = 0, 0
	rec_map = None

	if args.load_tree_sequence is not None:
		args.n_chr = 1

	m, m_geno = np.zeros(args.n_chr).astype(int), np.zeros(args.n_chr).astype(int)
	m_start, m_geno_start = np.zeros(args.n_chr).astype(int), np.zeros(args.n_chr).astype(int)

	return args, tree_sequence_list, tree_sequence_list_geno, m_total, m_geno_total, rec_map, m, m_start, m_geno, m_geno_start, genotyped_list_index

def get_common_mutations_ts(args, tree_sequence, log):

	common_sites = msprime.SiteTable()
	common_mutations = msprime.MutationTable()

	# Get the mutations > MAF.
	n_haps = tree_sequence.get_sample_size()
	log.log('Determining sites > MAF cutoff {m}'.format(m=args.maf))

	tables = tree_sequence.dump_tables()
	tables.mutations.clear()
	tables.sites.clear()

	for tree in tree_sequence.trees():
		for site in tree.sites():
			f = tree.get_num_leaves(site.mutations[0].node) / n_haps
			if f > args.maf and f < 1-args.maf:
				common_site_id = tables.sites.add_row(
					position=site.position,
					ancestral_state=site.ancestral_state)
				tables.mutations.add_row(
					site=common_site_id,
					node=site.mutations[0].node,
					derived_state=site.mutations[0].derived_state)
	new_tree_sequence = tables.tree_sequence()
	return new_tree_sequence

def set_mutations_in_tree(tree_sequence, p_causal):

	tables = tree_sequence.dump_tables()
	tables.mutations.clear()
	tables.sites.clear()
	
	causal_bool_index = np.zeros(tree_sequence.num_mutations, dtype=bool)
	# Get the causal mutations.
	k = 0
	for site in tree_sequence.sites():
		if np.random.random_sample() < p_causal:
			causal_bool_index[k] = True
			causal_site_id = tables.sites.add_row(
				position=site.position,
				ancestral_state=site.ancestral_state)
			tables.mutations.add_row(
				site=causal_site_id,
				node=site.mutations[0].node,
				derived_state=site.mutations[0].derived_state)
		k = k+1

	new_tree_sequence = tables.tree_sequence()
	m_causal = new_tree_sequence.num_mutations

	return new_tree_sequence, m_causal, causal_bool_index

def load_tree_sequence(args, log):

	# Create a list to fill with tree_sequences.
	args, tree_sequence_list, tree_sequence_list_geno, m_total, m_geno_total, rec_map, m, m_start, m_geno, m_geno_start, genotyped_list_index = initialise(args)
	tree_sequence_list.append(msprime.load(args.load_tree_sequence))
	args.n = int(tree_sequence_list[0].get_sample_size() / 2)
	N = args.n
	n_pops = 1

	log.log("Warning: load tree sequence was included for debugging, we don't support more than 1 population, and more than 1 chromosome.")

	common_mutations = []
	n_haps = tree_sequence_list[0].get_sample_size()

	log.log('n haplotypes read in: {n_haps}'.format(n_haps=n_haps))

	# Get the mutations > MAF.
	tree_sequence_list[0] = get_common_mutations_ts(args, tree_sequence_list[0], log)

	m[0] = int(tree_sequence_list[0].get_num_mutations())
	m_start[0] = 0
	m_total = m[0]
	log.log('Number of mutations above MAF in the generated data: {m}'.format(m=m[0]))
	log.log('Running total of sites > MAF cutoff: {m}'.format(m=m_total))

	# If genotyped proportion is < 1.
	if args.geno_prop is not None:
		tree_sequence_tmp, m_geno_tmp, genotyped_index = set_mutations_in_tree(tree_sequence_list[0], args.geno_prop)
		tree_sequence_list_geno.append(tree_sequence_tmp)
		genotyped_list_index.append(genotyped_index)
		m_geno[0] = int(m_geno_tmp)
		m_geno_start[0] = m_geno_total
		m_geno_total = m_geno[0]
		log.log('Number of sites genotyped in the generated data: {m}'.format(m=m_geno[0]))
		log.log('Running total of sites genotyped: {m}'.format(m=m_geno_total))
	else:
		tree_sequence_list_geno.append(tree_sequence_list[0])
		genotyped_list_index.append(np.ones(tree_sequence_list[0].num_mutations))
		m_geno[0] = m[0]
		m_geno_start[0] = m_start[0]
		m_geno_total = m_total
		log.log('Number of sites genotyped in the generated data: {m}'.format(m=m_geno[0]))
		log.log('Running total of sites genotyped: {m}'.format(m=m_geno_total))

	return tree_sequence_list, tree_sequence_list_geno, m, m_start, m_total, m_geno, m_geno_start, m_geno_total, N, n_pops, genotyped_list_index

def simulate_tree_sequence(args, rec_map_list, log):

	# Create a list to fill with tree_sequences.
	args, tree_sequence_list, tree_sequence_list_geno, m_total, m_geno_total, rec_map, m, m_start, m_geno, m_geno_start, genotyped_list_index = initialise(args)
	N = args.n

	# Choose the population demographic model to use.

	if args.sim_type == 'standard':
		migration_matrix = None
		population_configurations = None
		migration_matrix = None
		demographic_events = []
		sample_size = 2*args.n
		n_pops = 1

	if args.sim_type not in set(['standard', 'out_of_africa', 'out_of_africa_all_pops', 'unicorn']):
		raise Exception('Simulation type not found.')

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

	if args.sim_type != 'standard':
		dp = msprime.DemographyDebugger(Ne=args.Ne,
			population_configurations=population_configurations,
			migration_matrix=migration_matrix,
			demographic_events=demographic_events)
		dp.print_history()

	for chr in range(args.n_chr):
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
		tree_sequence_list[chr] = get_common_mutations_ts(args, tree_sequence_list[chr], log)

		m[chr] = int(tree_sequence_list[chr].get_num_mutations())
		m_start[chr] = m_total
		m_total += m[chr]
		log.log('Number of mutations above MAF in the generated data: {m}'.format(m=m[chr]))
		log.log('Running total of sites > MAF cutoff: {m}'.format(m=m_total))

		# If genotyped proportion is < 1.
		if args.geno_prop is not None:
			tree_sequence_tmp, m_geno_tmp, genotyped_index = set_mutations_in_tree(tree_sequence_list[chr], args.geno_prop)
			tree_sequence_list_geno.append(tree_sequence_tmp)
			genotyped_list_index.append(genotyped_index)
			m_geno[chr] = int(m_geno_tmp)
			m_geno_start[chr] = m_geno_total
			m_geno_total += m_geno[chr]
			log.log('Number of sites genotyped in the generated data: {m}'.format(m=m_geno[chr]))
			log.log('Running total of sites genotyped: {m}'.format(m=m_geno_total))
		else:
			tree_sequence_list_geno.append(tree_sequence_list[chr])
			genotyped_list_index.append(np.ones(tree_sequence_list[chr].num_mutations, dtype=bool))
			m_geno[chr] = m[chr]
			m_geno_start[chr] = m_start[chr]
			m_geno_total = m_total
			log.log('Number of sites genotyped in the generated data: {m}'.format(m=m_geno[chr]))
			log.log('Running total of sites genotyped: {m}'.format(m=m_geno_total))

		# Do you want to write the files to .vcf?
		# The trees shouldn't be written at this point if we're
		# using ascertained samples (which are always case control),
		# as it'll write ALL the samples to disk rather than just those that we sample.
		if args.write_trees and args.case_control is False:
			log.log('Writing genotyped information to disk')
			write.trees(args.out, tree_sequence_list_geno[chr], chr, m_geno[chr], n_pops, N, sim, args.vcf, np.arange(N))

	return tree_sequence_list, tree_sequence_list_geno, m, m_start, m_total, m_geno, m_geno_start, m_geno_total, N, n_pops, genotyped_list_index
