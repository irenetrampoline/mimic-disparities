bin/mallet import-dir --input data/notes/ --output notes.mallet --remove-stopwords
bin/mallet train-topics --input notes.mallet  --num-topics 50 --output-doc-topics data/mallet_topics_2500.txt --output-topic-keys data/mallet_topics_50_2500.txt --num-iterations 2500