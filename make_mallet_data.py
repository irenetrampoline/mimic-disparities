import pandas as pd

# read NEW adult notes, 15k vs 27348 admission over 21376 patients
notes_df = pd.read_csv('patients_notes.csv')

df = notes_df
for i in df.index:
    text = df.ix[i]['chartext']
    icuid = df.ix[i]['icustay_id']
    f = open('data/notes/%s.txt' % icuid, 'wb')
    f.write(text)
    f.close()