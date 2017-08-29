import math
import msprime

def out_of_africa(N_haps, no_migration):
	N_A = 7300
	N_B = 2100
	N_AF = 12300
	N_EU0 = 1000
	N_AS0 = 510
	# Times are provided in years, so we convert into generations.
	generation_time = 25
	T_AF = 220e3 / generation_time
	T_B = 140e3 / generation_time
	T_EU_AS = 21.2e3 / generation_time
	# We need to work out the starting (diploid) population sizes based on
	# the growth rates provided for these two populations
	r_EU = 0.004
	r_AS = 0.0055
	N_EU = N_EU0 / math.exp(-r_EU * T_EU_AS)
	N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)
	# Migration rates during the various epochs.

	if no_migration:
		m_AF_B = 0
		m_AF_EU = 0
		m_AF_AS = 0
		m_EU_AS = 0
	else:
		m_AF_B = 25e-5
		m_AF_EU = 3e-5
		m_AF_AS = 1.9e-5
		m_EU_AS = 9.6e-5
	
	# Population IDs correspond to their indexes in the population
	# configuration array. Therefore, we have 0=YRI, 1=CEU and 2=CHB
	# initially.
	n_pops = 3

	population_configurations = [
		msprime.PopulationConfiguration(sample_size=2*N_haps[0], initial_size=N_AF),
		msprime.PopulationConfiguration(sample_size=2*N_haps[1], initial_size=N_EU, growth_rate=r_EU),
		msprime.PopulationConfiguration(sample_size=2*N_haps[2], initial_size=N_AS, growth_rate=r_AS)
		]
	
	migration_matrix = [[0, m_AF_EU, m_AF_AS],
						[m_AF_EU, 0, m_EU_AS],
						[m_AF_AS, m_EU_AS, 0],
						]
	
	demographic_events = [
	# CEU and CHB merge into B with rate changes at T_EU_AS
	msprime.MassMigration(time=T_EU_AS, source=2, destination=1, proportion=1.0),
	msprime.MigrationRateChange(time=T_EU_AS, rate=0),
	msprime.MigrationRateChange(time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
	msprime.MigrationRateChange(time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
	msprime.PopulationParametersChange(time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
	# Population B merges into YRI at T_B
	msprime.MassMigration(time=T_B, source=1, destination=0, proportion=1.0),
	# Size changes to N_A at T_AF
	msprime.PopulationParametersChange(time=T_AF, initial_size=N_A, population_id=0)
	]
	# Return the output required for a simulation study.
	return population_configurations, migration_matrix, demographic_events, N_A, n_pops

def unicorn(N_haps):
	N_A = 7300
	N_B = 2100
	N_AF = 12300
	N_EU0 = 1000
	N_AS0 = 510
	# Times are provided in years, so we convert into generations.
	generation_time = 25
	T_AF = 220e3 / generation_time
	T_B = 140e3 / generation_time
	T_EU_AS = 21.2e3 / generation_time

	T_EU_1 = 10e3 / generation_time
	T_EU_2 = 5e3 / generation_time
	
	# We need to work out the starting (diploid) population sizes based on
	# the growth rates provided for these two populations
	r_EU = 0.004
	r_AS = 0.0055

	# Sanity checking:
	N_EU_1_0 = (N_EU0 /  math.exp(-r_EU * (T_EU_AS - T_EU_1))) * 0.5
	N_EU_1 = (N_EU_1_0 / math.exp(-r_EU * T_EU_1))
	N_EU_2_0 = (N_EU_1_0 / math.exp(-r_EU * (T_EU_1 - T_EU_2))) * 0.5
	N_EU_2 = (N_EU_2_0 / math.exp(-r_EU * T_EU_2))

	N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)
	
	# Population IDs correspond to their indexes in the population
	# configuration array. Therefore, we have 0=YRI, 1=CEU_1, 2=CEU_2, 3=CEU_3, and 4=CHB
	# initially.
	n_pops = 5

	population_configurations = [
		msprime.PopulationConfiguration(sample_size=2*N_haps[0], initial_size=N_AF),
		msprime.PopulationConfiguration(sample_size=2*N_haps[1], initial_size=N_EU_1, growth_rate=r_EU),
		msprime.PopulationConfiguration(sample_size=2*N_haps[2], initial_size=N_EU_2, growth_rate=r_EU),
		msprime.PopulationConfiguration(sample_size=2*N_haps[3], initial_size=N_EU_2, growth_rate=r_EU),
		msprime.PopulationConfiguration(sample_size=2*N_haps[4], initial_size=N_AS, growth_rate=r_AS)
		]
	
	migration_matrix = [[0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0]]
	
	demographic_events = [
	# CEU_1 and CEU_2 merge at T_EU_2
	msprime.MassMigration(time=T_EU_2, source=3, destination=2, proportion=1.0),
	# CEU_1_2 and CEU_3 merge at T_EU_1
	msprime.MassMigration(time=T_EU_1, source=2, destination=1, proportion=1.0),
	# CEU and CHB merge into B with rate changes at T_EU_AS
	msprime.MassMigration(time=T_EU_AS, source=4, destination=1, proportion=1.0),
	msprime.PopulationParametersChange(time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
	# Population B merges into YRI at T_B
	msprime.MassMigration(time=T_B, source=1, destination=0, proportion=1.0),
	# Size changes to N_A at T_AF
	msprime.PopulationParametersChange(time=T_AF, initial_size=N_A, population_id=0)
	]
	# Return the output required for a simulation study.
	return population_configurations, migration_matrix, demographic_events, N_A, n_pops


def migration(N_haps):
	# M is the overall symmetric migration rate, d is the number of demes.
	M = 0.2
	d = 2
	# We rescale m into per-generation values for msprime.
	m = M / (4 * (d - 1))
	# Allocate the initial sample.
	population_configurations = [
	msprime.PopulationConfiguration(sample_size=2*N_haps[0]),
	msprime.PopulationConfiguration(sample_size=2*N_haps[1])
	]
	# Now we set up the migration matrix.
	# This is a symmetric island model, so we have the same rate of migration 
	# between all pairs of demes. Diagonal elements must be zero.
	migration_matrix = [[0, m],[m, 0]]
	# We pass these values to the simulate function.
	return(population_configurations, migration_matrix)
