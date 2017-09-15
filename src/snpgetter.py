from __future__ import division
import numpy as np

def nextSNP(variant, index=None):
	if index is None:
		var_tmp = np.array(map(int, variant.genotypes[0::2])) + np.array(map(int, variant.genotypes[1::2]))
	else:
		var_tmp = np.array(map(int, variant.genotypes[0::2][index])) + np.array(map(int, variant.genotypes[1::2][index]))
	
	n = len(var_tmp)
	# Additive term.
	mean_X = np.mean(var_tmp)
	p = mean_X / 2
	# Evaluate the mean and then sd to normalise.
	X_A = (var_tmp - mean_X) / np.std(var_tmp)
	# Dominance term.
	X_D = np.ones(n)
	X_D[var_tmp == 0] = - p / (1 - p)
	X_D[var_tmp == 2] = - (1 - p) / p
	# Evaluate the mean and then sd to normalise.
	X_D = (X_D - np.mean(X_D)) / np.std(X_D)
	return X_A, X_D

def nextSNP_add(variant, index=None):
	
	if index is None:
		var_tmp = np.array(map(int, variant.genotypes[0::2])) + np.array(map(int, variant.genotypes[1::2]))
	else:
		var_tmp = np.array(map(int, variant.genotypes[0::2][index])) + np.array(map(int, variant.genotypes[1::2][index]))

	# Additive term.
	mean_X = np.mean(var_tmp)
	p = mean_X / 2
	# Evaluate the mean and then sd to normalise.
	X_A = (var_tmp - mean_X) / np.std(var_tmp)
	return X_A
	