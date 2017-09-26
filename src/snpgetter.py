from __future__ import division
import numpy as np

# Normalize additive and dominance entries in sample.
def standardize(X):
	n = len(X)
	mean_X = np.mean(X)
	p = mean_X / 2
	X_A = (X - mean_X) / np.std(X)

	X_D = np.copy(X)
	X_D[X_D == 2] = 0

	G = np.concatenate((np.ones((n, 1)), X_A[:, None], X_D[:, None]), axis = 1)
	X_D = math.sqrt(n) * np.linalg.qr(G)[0][:,-1]

	# X_D = np.ones(n)
	# X_D[X == 0] = - p / (1 - p)
	# X_D[X == 2] = - (1 - p) / p
	# X_D = (X_D - np.mean(X_D)) / np.std(X_D)
	
	return X_A, X_D

def nextSNP(variant, Xs, index=None):
	if index is None:
		var_tmp = np.array(map(int, variant.genotypes[0::2])) + np.array(map(int, variant.genotypes[1::2]))
	else:
		var_tmp = np.array(map(int, variant.genotypes[0::2][index])) + np.array(map(int, variant.genotypes[1::2][index]))
	
	n = len(var_tmp)
	X_A, X_D = standardize(var_tmp)

	X_E = np.zeros(n)

	if Xs is not None:
		Xs_A, Xs_D = standardize(Xs)
		G = np.concatenate((np.ones((n, 1)), X_A[:, None], Xs_A[:, None], X_D[:, None], Xs_D[:, None], np.multiply(X_A, Xs_A)[:, None]), axis=1)

		X_E = math.sqrt(n) * np.linalg.qr(G)[0][:, -1]

	return X_A, X_D, X_E

def nextSNP_add(variant, Xs, index=None):
	X_A, X_D, X_E = nextSNP(variant, Xs, index=None)
	return X_A

def nextSNP_dom(variant, Xs, index=None):
	X_A, X_D, X_E = nextSNP(variant, Xs, index=None)
	return X_D

def nextSNP_epi(variant, Xs, index=None):
	X_A, X_D, X_E = nextSNP(variant, Xs, index=None)
	return X_E
	