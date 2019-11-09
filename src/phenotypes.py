from __future__ import division
import msprime
import numpy as np
import random
import tqdm
import scipy.stats as sp
import src.regressions as reg
import src.tools as tl
import src.tree_sequence as ts
import src.snpgetter as sg
import src.ldscores as ld
import src.printing as pr
import statsmodels.api as sm
import time, sys, traceback, argparse
import src.write as write
import pandas as pd

def case_control(y, prevalence, sample_prevalence, N):
        # Determine the liability threshold.
        p = prevalence
        T = sp.norm.ppf(1-p)
        
        # Index the cases.
        cases = [i for (i, x) in enumerate(y) if x >= T]
        mask = np.ones(len(y), dtype=bool)
        mask[cases] = False
        n_cases = len(cases)

        if sample_prevalence is None:
                n_controls = N - n_cases
        else:
                n_controls = int(((1-sample_prevalence) / sample_prevalence) * n_cases)

        controls = np.arange(N)
        if (N - n_cases) < n_controls:
                n_controls = N - n_cases
                log.log('Warning: this condition should not hold - '
                        'is sample prevalence close to population prevalence?')
                controls = controls[mask]
        else:
                controls = controls[mask][random.sample(range(N - n_cases), n_controls)]

        controls = sorted(controls)
        return cases, controls, n_cases, n_controls, T

def get_phenotypes(args, N, n_pops, tree_sequence_list, m_total, log):

        y = np.zeros(N)

        if args.debug:
                random.seed(1)
                np.random.seed(1)

        if args.C_bool:
                C = np.random.binomial(1, args.C_bool_p, size=N)
                C = (C - np.mean(C)) / np.std(C)
                log.log('Boolean covariate.')
                log.log('Average kurtosis for these phenotypes: {K}.'.format(K=np.sum(C**4)/N))
        else:
                C = np.random.normal(loc=0, scale=1, size=N)
                C = (C - np.mean(C)) / np.std(C)
                # print the average kurtosis across the individuals
                log.log('Normally distributed covariate. Kurtosis should be around 3.')
                log.log('Average kurtosis for these phenotypes: {K}.'.format(K=np.sum(C**4)/N))
        
        if args.include_pop_strat is True and args.s2 > 0:
                # Get the means for the populations.
                alpha = np.random.normal(loc=0, scale=np.sqrt(args.s2), size=n_pops)
                log.log(alpha)
                # Add pop-strat additions to the phenotype vector, conditional on the population sampled from.
                for pops in range(n_pops):
                        pop_leaves = tree_sequence_list[0].get_samples(population_id=pops)
                        len(map(int, [x/2 for x in pop_leaves[0::2]]))
                        y[map(int, [x/2 for x in pop_leaves[0::2]])] += alpha[pops]

        for chr in range(args.n_chr):
                m_chr = int(tree_sequence_list[chr].get_num_mutations())
                log.log('Picking causal variants and determining effect sizes in chromosome {chr}'.format(chr=chr+1))
                log.log('p-causal is {p_causal}'.format(p_causal=args.p_causal))
                
                if (((1 + int(args.dominance) + int(args.gxe)) * args.p_causal) < 1) or args.same_causal_sites: # If the number of runs through the data is less than 1, run this speedup.
                        log.log('Running speedup to ensure fewer runs through the data')
                        tree_sequence_pheno_A, m_causal_A, causal_A_index = ts.set_mutations_in_tree(tree_sequence_list[chr], args.p_causal)
                        log.log('Picked {m} additive causal variants out of {mc}'.format(m=m_causal_A, mc=m_chr))

                        if args.same_causal_sites is False:
                                log.log('Different causal sites for other heritability contributions')
                                tree_sequence_pheno_D, m_causal_D, causal_D_index = ts.set_mutations_in_tree(tree_sequence_list[chr], args.p_causal)
                                if args.h2_D > 0: log.log('Picked {m} dominance causal variants out of {mc}'.format(m=m_causal_D, mc=m_chr))
                                tree_sequence_pheno_AC, m_causal_AC, causal_AC_index = ts.set_mutations_in_tree(tree_sequence_list[chr], args.p_causal)
                                if args.h2_AC > 0: log.log('Picked {m} gxe causal variants out of {mc}'.format(m=m_causal_AC, mc=m_chr))

                                if args.h2_A > 0:
                                        beta_A = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_total * args.p_causal)), size=m_causal_A)
                                        # Get the phenotypes.
                                        k = 0
                                        log.log('Determining phenotype data: additive.')
                                        for variant in tl.progress(args.progress_bars, tree_sequence_pheno_A.variants(), total=m_causal_A): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
                                                X_A = sg.nextSNP_add(variant)
                                                # Effect size on the phenotype.
                                                y += X_A * beta_A[k]
                                                k += 1

                                if args.dominance and args.h2_D >0:
                                        beta_D = np.random.normal(loc=0, scale=np.sqrt(args.h2_D / (m_total * args.p_causal)), size=m_causal_D)
                                        k = 0
                                        log.log('Determining phenotype data: dominance.')
                                        for variant in tl.progress(args.progress_bars, tree_sequence_pheno_D.variants(), total=m_causal_D): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
                                                X_A, X_D = sg.nextSNP(variant)
                                                # Effect size on the phenotype.
                                                y += X_D * beta_D[k]
                                                k += 1

                                if args.gxe and args.h2_AC > 0:
                                        beta_AC = np.random.normal(loc=0, scale=np.sqrt(args.h2_AC / (m_total * args.p_causal)), size=m_causal_AC)
                                        # If examining interaction with a covariate, pick the values of the covariate, and normalise.
                                        k = 0
                                        log.log('Determining phenotype data: gene x environment.')
                                        for variant in tl.progress(args.progress_bars, tree_sequence_pheno_AC.variants(), total=m_causal_AC): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
                                                X_A = sg.nextSNP_add(variant)
                                                # Effect size on the phenotype.
                                                y += C * X_A * beta_AC[k]
                                                k += 1
                        else:
                                log.log('Same causal sites for other heritability contributions')
                                beta_A, beta_D, beta_AC = np.zeros(m_causal_A), np.zeros(m_causal_A), np.zeros(m_causal_A)  
                                if args.h2_A > 0:
                                        beta_A = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_total * args.p_causal)), size=m_causal_A)
                                
                                if args.dominance and args.h2_D > 0:
                                        beta_D = np.random.normal(loc=0, scale=np.sqrt(args.h2_D / (m_total * args.p_causal)), size=m_causal_A)

                                if args.gxe and args.h2_AC > 0:
                                        beta_AC = np.random.normal(loc=0, scale=np.sqrt(args.h2_AC / (m_total * args.p_causal)), size=m_causal_A)

                                k = 0
                                log.log('Determining phenotype data')

                                # Note that we use just one tree_sequence here, because the causal sites are the same in this portion of the code.
                                for variant in tl.progress(args.progress_bars, tree_sequence_pheno_A.variants(), total=m_causal_A): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
                                        X_A, X_D = sg.nextSNP(variant)
                                        # Effect size on the phenotype.
                                        y += X_A * beta_A[k] + X_D * beta_D[k] + C * X_A * beta_AC[k]
                                        k += 1

                else:
                        m_causal = int(m_chr * args.p_causal)
                        beta_A, beta_D, beta_AC = np.zeros(m_chr), np.zeros(m_chr), np.zeros(m_chr)
                        beta_A_causal_index = random.sample(range(m_chr), m_causal)
                        log.log('Picked {m} additive causal variants out of {mc}'.format(m=m_causal, mc=m_chr))

                        if args.h2_A > 0:
                                beta_A[beta_A_causal_index] = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_total * args.p_causal)), size=m_causal)
                        
                        if args.dominance:
                                beta_D, beta_D_causal_index = np.zeros(m_chr), random.sample(range(m_chr), m_causal)
                                log.log('Picked {m} dominance causal variants out of {mc}'.format(m=m_causal, mc=m_chr))
                                if args.h2_D > 0:
                                        beta_D[beta_D_causal_index] = np.random.normal(loc=0, scale=np.sqrt(args.h2_D / (m_total * args.p_causal)), size=m_causal)

                        if args.gxe:
                                beta_AC, beta_AC_causal_index = np.zeros(m_chr), random.sample(range(m_chr), m_causal)
                                log.log('Picked {m} gxe causal variants out of {mc}'.format(m=m_causal, mc=m_chr))
                                if args.h2_AC > 0:
                                        beta_AC[beta_AC_causal_index] = np.random.normal(loc=0, scale=np.sqrt(args.h2_AC / (m_total * args.p_causal)), size=m_causal)
                        
                        # Get the phenotypes.
                        k = 0
                        log.log('Determining phenotype data.')

                        for variant in tl.progress(args.progress_bars, tree_sequence_list[chr].variants(), total=m_chr): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
                                X_A, X_D = sg.nextSNP(variant)
                                # Effect size on the phenotype.
                                y += X_A * beta_A[k] + X_D * beta_D[k] + X_A * C * beta_AC[k]
                                k += 1

                if args.write_betas:
                        np.savetxt(fname=args.out+'.true_beta_A.chr'+str(chr+1)+'.tsv',
                                   X=beta_A,
                                   fmt='%.3e',
                                   delimiter='\t',
                                   header=args.out+'.true_beta_A.chr'+str(chr+1))
        # Add noise to the y.
        y += np.random.normal(loc=0, scale=np.sqrt(1-(args.h2_A+args.h2_D+args.h2_AC+args.s2)), size=N)
        # Finally, normalise.
        y = (y - np.mean(y)) / np.std(y)

        return y, C

# Here, want to create a chi sq function, and an LD score function.

def get_chisq(args, tree_sequence_list_geno, m_geno, m_geno_total, y, N, C, sim, log):
        # Initialise the chi squared statistics.
        chisq_A, chisq_D, chisq_AC = np.zeros((m_geno_total,1)), np.zeros((m_geno_total,1)), np.zeros((m_geno_total,1))

        if args.case_control:
                log.log("Running case-control simulation.")
                if args.prevalence is None:
                        raise ValueError("prevalence must be set if running case-control analysis.")
                cases, controls, n_cases, n_controls, T = case_control(y, args.prevalence, args.sample_prevalence, N)
                n = n_cases + n_controls
                log.log('Ascertaining for sample prevalence {s} from population prevalence {p}: {N} total samples'.format(s=args.sample_prevalence, p=args.prevalence, N=n))
                y_cc = np.zeros(n)
                y_cc[:n_cases] = 1
                index = cases + controls
                C_sim = C[index]

                if args.linear is False and args.ldsc is True:
                        k = 0
                        for chr in range(args.n_chr):
                                for variant in tl.progress(args.progress_bars, tree_sequence_list_geno[chr].variants(), total=m_geno[chr]):
                                        X_A, X_D = sg.nextSNP(variant, index = index)
                                        chisq_A[k] = sm.Logit(y_cc, sm.add_constant(X_A)).fit(disp=0).llr
                                        chisq_D[k] = sm.Logit(y_cc, sm.add_constant(X_D)).fit(disp=0).llr
                                        chisq_AC[k] = sm.Logit(y_cc, sm.add_constant(C_sim * X_A)).fit(disp=0).llr
                                        k += 1

        if ( ((args.case_control is False) or (args.case_control is True and args.linear is True)) and args.ldsc is True ):
                if args.case_control:
                        log.log("Warning: running linear regression for case-control.")
                        y = (y_cc - np.mean(y_cc)) / np.std(y_cc)
                        index = cases + controls
                        C_sim = C[index]
                else:
                        index = None
                        C_sim = C
                        n = N

                # Then use these ys to determine beta hats.
                if args.write_betas:
                        k_beta = 0 # index for SNP beta
                        intercept = np.ones(shape=(1,n)) # veector of intercepts for least sq linear regression
                        for chr in range(args.n_chr):
                                beta_A = np.empty(shape=m_geno_total)
                                log.log('Determining beta-hats in chromosome {chr}'.format(chr=chr+1))
                                for variant in tree_sequence_list_geno[chr].variants():
                                        X_A, X_D = sg.nextSNP(variant, index=index)
                                        X_A_w_int = np.vstack((X_A.reshape(1, n), intercept)).T
                                        coef, _, _, _ = np.linalg.lstsq(X_A_w_int, y.reshape(n,),rcond=None)
                                        beta_A[k_beta] = coef[0] # take only the beta for the genotypes, not the intercept
                                        k_beta += 1
                                np.savetxt(fname=args.out+'.marginal_beta_A.chr'+str(chr+1)+'.tsv', 
                                           X=beta_A,
                                           fmt='%.3e',
                                           delimiter='\t',
                                           header=args.out+'.marginal_beta_A.chr'+str(chr+1))
                k = 0
                for chr in range(args.n_chr):
                        log.log('Determining chi-squared statistics in chromosome {chr}'.format(chr=chr+1))
                        for variant in tree_sequence_list_geno[chr].variants():
                                X_A, X_D = sg.nextSNP(variant, index=index)
                                # Then sum to get the effect size on the phenotype.
                                chisq_A[k] = np.dot(y.reshape(1,n), X_A)**2 / n
                                chisq_D[k] = np.dot(y.reshape(1,n), X_D)**2 / n
                                chisq_AC[k] = np.dot(y.reshape(1,n), C_sim * X_A)**2 / n
                                k += 1
        if args.write_pheno or args.write_trees:
                if args.case_control:
                        sample_ID = index
                        y = y_cc.astype(int)
                        if args.write_trees:
                                tree_index = [[2*x,2*x+1] for x in index]
                                tree_index = [j for x in tree_index for j in x]

                                for chr in range(args.n_chr):
                                        tree_sequence_to_write = tree_sequence_list_geno[chr].simplify(tree_index)
                                        write.trees(args.out, tree_sequence_to_write, chr, m_geno[chr], n_pops, N, sim, args.vcf, index)

                else:
                        sample_ID = np.arange(N)
                df_pheno=pd.DataFrame({'sample_ID':sample_ID, 'phenotype':y})
                df_pheno.to_csv(args.out + ".sim" + str(sim+1) + '.pheno.tsv', sep='\t', header=True, index=False)

        if args.case_control:
                return chisq_A, chisq_D, chisq_AC, n, C_sim, index, y_cc, n_cases, T
        else:
                return chisq_A, chisq_D, chisq_AC, n, C_sim, index
