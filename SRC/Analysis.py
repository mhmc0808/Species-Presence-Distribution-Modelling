#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 14:55:33 2025

@author: zacharygavin
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


########################################################################################
############# ------------------------------------------------------------##############
# ---------------------------- Pseudo-absence generation ----------------------------- #
############# ------------------------------------------------------------##############
########################################################################################

########################################################################
############# ---------- Gradient boosting ------------- ###############
########################################################################

# ------------------ TUNED ------------------

onevr_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/GB tuned onevrest accuracy.csv")
unif_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/GB tuned psuedo abesence accuracy.csv")


# ------------------ BASE ------------------


onevr_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/GB base onevrest accuracy.csv")
unif_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/GB base psuedo abesence accuracy.csv")


# ------------ PAIRED WILCOXON RANK SUM TESTING -----------

analysis_df = onevr_df[['Species ID', 'auc_roc']]
analysis_df['auc_unif'] = unif_df['auc_roc']
analysis_df = analysis_df.dropna()


# 33/41, 31/45 species have higher ROC from one vs rest pseudo-absences
sum(analysis_df['auc_roc']>analysis_df['auc_unif'])


# Do hypothesis test to see if there is a statistically sig. difference between dists
# Use Wilcoxon and boxplots are pretty clearly not normal


res = wilcoxon(analysis_df['auc_roc'], analysis_df['auc_unif'])

res.statistic
res.pvalue

np.mean(analysis_df['auc_roc'])
np.median(analysis_df['auc_roc'])

########################################################################
############# ---------- Random Forest ------------- ################
########################################################################

# ------------------ TUNED ------------------

onevr_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/RF Tuned onevrest accuracy.csv")
unif_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/RF Tuned psuedo abesence accuracy.csv")


# ------------------ BASE ------------------


onevr_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/RF Base onevrest accuracy.csv")
unif_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/RF Base psuedo abesence accuracy.csv")


# ------------ PAIRED WILCOXON RANK SUM TESTING -----------

analysis_df = onevr_df[['Species ID', 'auc_roc']]
analysis_df['auc_unif'] = unif_df['auc_roc']
analysis_df = analysis_df.dropna()


# 26/41, 29/41 species have higher ROC from one vs rest pseudo-absences
sum(analysis_df['auc_roc']>analysis_df['auc_unif'])


# Do hypothesis test to see if there is a statistically sig. difference between dists
# Use Wilcoxon and boxplots are pretty clearly not normal


res = wilcoxon(analysis_df['auc_roc'], analysis_df['auc_unif'], alternative='greater')

res.statistic
res.pvalue

np.mean(analysis_df['auc_roc'])
np.median(analysis_df['auc_roc'])


########################################################################
############# ---------- Logistic Regression------------- ##############
########################################################################

# ------------------ TUNED ------------------

onevr_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/LR Tuned onevrest accuracy.csv")
unif_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/LR Tuned psuedo abesence accuracy.csv")


# ------------------ BASE ------------------


onevr_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/LR Base onevrest accuracy.csv")
unif_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/LR Base psuedo abesence accuracy.csv")


# ------------ PAIRED WILCOXON RANK SUM TESTING -----------

analysis_df = onevr_df[['Species ID', 'auc_roc']]
analysis_df['auc_unif'] = unif_df['auc_roc']
analysis_df = analysis_df.dropna()


# 23/41, 28/41 species have higher ROC from one vs rest pseudo-absences
sum(analysis_df['auc_roc']>analysis_df['auc_unif'])


# Do hypothesis test to see if there is a statistically sig. difference between dists
# Use Wilcoxon and boxplots are pretty clearly not normal


res = wilcoxon(analysis_df['auc_roc'], analysis_df['auc_unif'])

res.statistic
res.pvalue




########################################################################################
############# ------------------------------------------------------------##############
# ------------------------------ Hyper-parameter tuning ------------------------------ #
############# ------------------------------------------------------------##############
########################################################################################


########################################################################
############# ---------- Gradient boosting ------------- ###############
########################################################################

# ------------------ OneVRest ------------------

base_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/GB tuned onevrest accuracy.csv")
tuned_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/GB base onevrest accuracy.csv")


# ------------------ Uniform ------------------


base_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/GB tuned psuedo abesence accuracy.csv")
tuned_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/GB base psuedo abesence accuracy.csv")


# ------------ PAIRED WILCOXON RANK SUM TESTING -----------

analysis_df = base_df[['Species ID', 'auc_roc']]
analysis_df['auc_tuned'] = tuned_df['auc_roc']
analysis_df = analysis_df.dropna()


# 33/41, 31/45 species have higher ROC from one vs rest pseudo-absences
sum(analysis_df['auc_roc']>analysis_df['auc_tuned'])


# Do hypothesis test to see if there is a statistically sig. difference between dists
# Use Wilcoxon and boxplots are pretty clearly not normal


res = wilcoxon(analysis_df['auc_roc'], analysis_df['auc_tuned'], alternative='greater')

res.statistic
res.pvalue

np.median(analysis_df['auc_roc']-analysis_df['auc_tuned'])
# 1vR: 0.00912
# unif:  0.0157




########################################################################
############# ---------- Random Forest ------------- ################
########################################################################

# ------------------ OneVRest ------------------

base_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/RF Tuned onevrest accuracy.csv")
tuned_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/RF Base onevrest accuracy.csv")


# ------------------ Uniform ------------------


base_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/RF Tuned psuedo abesence accuracy.csv")
tuned_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/RF Base psuedo abesence accuracy.csv")


# ------------ PAIRED WILCOXON RANK SUM TESTING -----------

analysis_df = base_df[['Species ID', 'auc_roc']]
analysis_df['auc_tuned'] = tuned_df['auc_roc']
analysis_df = analysis_df.dropna()


# 33/41, 31/45 species have higher ROC from one vs rest pseudo-absences
sum(analysis_df['auc_roc']>analysis_df['auc_tuned'])


# Do hypothesis test to see if there is a statistically sig. difference between dists
# Use Wilcoxon and boxplots are pretty clearly not normal


res = wilcoxon(analysis_df['auc_roc'], analysis_df['auc_tuned'], alternative='greater')

res.statistic
res.pvalue

np.median(analysis_df['auc_roc']-analysis_df['auc_tuned'])
# 1vR: 0.0176
# unif: 0.0195


########################################################################
############# ---------- Logistic Regression------------- ##############
########################################################################

# ------------------ OneVRest ------------------

base_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/LR Tuned onevrest accuracy.csv")
tuned_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/LR Base onevrest accuracy.csv")


# ------------------ Uniform ------------------


base_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/LR Tuned psuedo abesence accuracy.csv")
tuned_df = pd.read_csv("/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/LR Base psuedo abesence accuracy.csv")


# ------------ PAIRED WILCOXON RANK SUM TESTING -----------

analysis_df = base_df[['Species ID', 'auc_roc']]
analysis_df['auc_tuned'] = tuned_df['auc_roc']
analysis_df = analysis_df.dropna()


# 33/41, 31/45 species have higher ROC from one vs rest pseudo-absences
sum(analysis_df['auc_roc']>analysis_df['auc_tuned'])


# Do hypothesis test to see if there is a statistically sig. difference between dists
# Use Wilcoxon and boxplots are pretty clearly not normal


res = wilcoxon(analysis_df['auc_roc'], analysis_df['auc_tuned'], alternative='greater')

res.statistic
res.pvalue

np.median(analysis_df['auc_roc']-analysis_df['auc_tuned'])
# 1vR: 0.0405
# unif: 0.0815

# average of all differences: 0.031

