import pandas as pd

def get_mimic_notes(races = ['eth_white', 'eth_black'], all_races=False):
    if not all_races:
        race_names = [i.replace('eth_', '') for i in races]
        race_label = race_names[0] + '-OR-' + race_names[1]
        # options: eth_white, eth_black, eth_hispanic, eth_asian, eth_other
        df = pd.read_csv('data/patients_notes.csv')

        ignore_cols = set([
           u'mort_hosp', 'eth_asian', 'eth_black', 
           'eth_hispanic', 'eth_other', 'eth_white'])

        feature_cols = 'chartext'
        # feature_cols = [i for i in df.columns if i not in ignore_cols]

        target_col = 'mort_hosp'

        race_df = df[(df[races[0]] == 1) | (df[races[1]] == 1)]
        sub_df = race_df[['chartext',target_col]]

        sub_df[race_label] = df[races[1]] == 1
        sub_df[race_label] = sub_df[race_label].apply(int)

        # feature_cols += 'is_black'
        
        # pdb.set_trace()
        return sub_df, 'chartext', 'mort_hosp', race_label

    elif all_races:
        races_lst = ['eth_asian', 'eth_black', 
           'eth_hispanic', 'eth_other', 'eth_white']
        feature_col = 'chartext'
        target_col = 'mort_hosp'

        df = pd.read_csv('data/patients_notes.csv')

        def insur_pubpriv(x):
          if x == 'Medicare' or x == 'Medicaid' or x == 'Government':
              return 'public'
          elif x == 'Private':
              return 'private'
          elif x == 'Self Pay':
              return 'other'
              
        df['insur_group'] = df['insurance'].apply(insur_pubpriv)
        
        dummies = pd.get_dummies(df[['insur_group']])
        df[dummies.columns] = dummies
        df = df.drop('insur_group_other', axis=1)

        df['male'] = df['gender']
        df['female'] = 1 - df['gender']

        # sub_df = df[[feature_col, target_col] + races_lst]
        return df, 'chartext', 'mort_hosp', races_lst