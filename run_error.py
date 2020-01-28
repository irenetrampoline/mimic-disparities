import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pickle as pk

from collections import defaultdict
from itertools import combinations

from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from load_mimic import get_mimic_notes

def run_all_groups(trials=5, error='01', output_dir='results/'):
    data, feat, targ, race_cols = get_mimic_notes(all_races=True)

    models = ['LR']
    
    # race_cols = prot
    insur_cols = ['insur_group_private', 'insur_group_public']
    gender_cols = ['female', 'male']
    # for mod in models:
    for prot in [insur_cols, gender_cols, race_cols]:
        run_model(data, feat, targ, prot, trials, model='LR', error=error, output_dir=output_dir)

def run_model(data, feat, targ, prot, trials=5, model='LR', error='01', output_dir='results/'):
    """
    INPUTS
    -----
    data (pd.DataFrame)
    feat (str): output value
    options for model: ['RF', 'LR']
    """
    if 'eth_asian' in prot:
        exp_type = 'race'
    elif 'male' in prot:
        exp_type = 'gender'
    else:
        exp_type = 'insurance'
        
    if model == 'RF':
        clf = RandomForestClassifier(max_depth=10)
    elif model == 'LR':
        clf = LogisticRegression(penalty='l1', C=1., solver='liblinear')

    vect = TfidfVectorizer(max_features=10000)
    # trials = 5

    zo_results = defaultdict(list)
    fp_results = defaultdict(list)
    fn_results = defaultdict(list)

    for t in range(trials):
        print('trial', t)
        train_data, test_data = train_test_split(data, train_size=0.8)

        X_train, y_train = train_data[feat], train_data[targ]
        X_train = vect.fit_transform(X_train)

        clf.fit(X_train, y_train)

        X_test, y_test = test_data[feat], test_data[targ]
        X_test = vect.transform(X_test)

        test_N = dict()
        X_test_N, y_test_N = dict(), dict()
        for p in prot:
            test_N[p] = test_data[test_data[p]==1]
            X_test_N[p] = test_N[p][feat]
            y_test_N[p] = test_N[p][targ]

        for p in prot:
            X_test_N[p] = vect.transform(test_N[p][feat].values)

        acc = list()
        protected = prot
        pred_probaN = dict()
        for p in protected:
            pred_probaN[p] = clf.predict_proba(X_test_N[p])[:,1]
        comboN = dict()
        for p in protected:
            comboN[p] = np.vstack([y_test_N[p], pred_probaN[p]])

        i_Y0_AN = dict()
        i_Y1_AN = dict()
        FP_N = dict()
        FN_N = dict()
        ZO_N = dict()

        for p in protected: 
            i_Y0_AN[p] = np.where(y_test_N[p] == 0)[0]
            i_Y1_AN[p] = np.where(y_test_N[p] == 1)[0]

        for p in protected:
            # pdb.set_trace()
            FP_N[p] = 1./ len(i_Y0_AN[p]) * sum(pred_probaN[p][i_Y0_AN[p]])
            FN_N[p] = 1./ len(i_Y1_AN[p]) * sum(1 - pred_probaN[p][i_Y1_AN[p]])
            ZO_N[p] = mae(y_test_N[p], pred_probaN[p])

        
        for p in protected:
            zo_results[p] = zo_results[p] + list(abs(y_test_N[p] - pred_probaN[
                p]).values)
            # pdb.set_trace()
            # wherey0 = np.where(y_test_N[p] == )
            fp_results[p] = fp_results[p] + list(pred_probaN[p][i_Y0_AN[p]])
            fn_results[p] = fn_results[p] + list(1 - pred_probaN[p][i_Y1_AN[p]])

        if error == '01_old':
            for p in protected:
                zo_results[p] = clf.score(X_test_N[p], y_test_N[p])

    # ZERO ONE ERROR
    means = [np.mean(zo_results[p]) for p in protected]
    anova = stats.f_oneway(*[zo_results[i] for i in protected])

    df_lst = list()
    for p in protected:
        df = pd.DataFrame({'zo': zo_results[p]})
        df['race'] = p.replace('eth_', '').replace('insur_group_', '').replace('gender_', '').title()
        df_lst.append(df)
        
    zo_df = pd.concat(df_lst)

#     tukey = pairwise_tukeyhsd(groups=zo_df['race'].values, endog=zo_df['zo'].values)

    fname = exp_type

    # FALSE POSITIVE
    df_lst = list()
    for p in protected:
        df = pd.DataFrame({'fp': fp_results[p]})
        df['race'] = p.replace('eth_', '').replace('insur_group_', '').replace('gender_', '').title()
        df_lst.append(df)
        
    fp_df = pd.concat(df_lst)
#     tukey = pairwise_tukeyhsd(groups=fp_df['race'].values, endog=fp_df['fp'].values)

    fname = exp_type
    # model = 'RFC'
    # error = '01'
    
#     plot_sim2(tukey, fname='%s_%s_%s' % (fname, model, error), xlabel='False positive rate')

    # FALSE NEGATIVE
    df_lst = list()
    for p in protected:
        df = pd.DataFrame({'fn': fn_results[p]})
        df['race'] = p.replace('eth_', '').replace('insur_group_', '').replace('gender_', '').title()
        df_lst.append(df)
        
    fn_df = pd.concat(df_lst)

    fname = exp_type
    
    error = 'all'
    f = open('%s%s_%s_%s.pk' % (output_dir, fname, model, error), 'wb')
    cache = {'zo_results': zo_df, 'fp_results': fp_df, 'fn_results': fn_df, 
            'groups': protected}
    pk.dump(cache,f)
    f.close()

    print('saved to','%s%s_%s_%s.pk' % (output_dir, fname, model, error))