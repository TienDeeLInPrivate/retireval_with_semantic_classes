import pandas as pd

file = "Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/trec/data/pubmed/result_hybrid_pubmed_scores.pkl"
df = pd.read_pickle(file)
#df = pd.read_csv(file)

df['old_true_passage_position'] = 101  

for index, row in df.iterrows():

    retrieved_list = row['retrieved_hash_pos_in_doc']

    if row['true_doc_hash_pos_in_doc'] in retrieved_list:
        print('true')
        pos = retrieved_list.index(row['true_doc_hash_pos_in_doc']) 
        df.at[index, 'old_true_passage_position'] = pos + 1
    else:
        df.at[index, 'old_true_passage_position'] = 101

print(df)

df.to_pickle(file)
df.to_csv("Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/trec/data/pubmed/result_hybrid_pubmed_scores.csv")
