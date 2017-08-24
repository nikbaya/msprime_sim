# msprime sim

*A collection of simulation schemes and methods of anaysis - extensions of LD-score and PCGC.*

The simulation portion of the code allows users to easily generate genotype data under a flexible collection of demographic scenarios and, given this genotype data, generate phenotype information under simple models mapping genotype to phenotype.

Simulation of genetic data wraps the ``msprime`` library which allows extremely efficient sampling from the coalescent with recombination. See the associated [paper](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004842) and [repository](https://github.com/jeromekelleher/msprime).

Note: the code presented here requires the latest and greatest version of ``msprime``: the commit at the time of writing this README was ``dbec779``.

The initial goal of this simulation code was to test extensions of [LD-score](http://www.nature.com/ng/journal/v47/n3/full/ng.3211.html) and [PCGC](http://www.pnas.org/content/111/49/E5272.short). As such, flags can be used to simulate additive, dominance, and gene by environment contributions to phenotype. Further, extensions to the LD-score and PCGC software are implemented here and can be used to attempt to recapitulate these contributions. Details of these extensions can be found below and in the [write-up](https://github.com/astheeggeggs/ldscgxe/tree/master/writeup) of the [``ldscgxe``](https://github.com/astheeggeggs/ldscgxe) repository.

More recently, we have used the wrapper around msprime to simulate demographic scenarios and associated phenotype data to test other methods currently under development in the Neale lab.

We are hopeful that the general simulation schemes provided here may be of use to the human genetics community more broadly.

The code consists of two components, simulation and analysis.

## Simulation
Flags relate to the simulation of genotype information (which calls ``msprime``) and phenotype information, which, conditional on genotype information, determines phenotype data under the specified model.

### Genotype
Include the options with extended descriptions.
### Phenotype
Include the options with extended descriptions.

## Analysis
Include the options with extended descriptions.
