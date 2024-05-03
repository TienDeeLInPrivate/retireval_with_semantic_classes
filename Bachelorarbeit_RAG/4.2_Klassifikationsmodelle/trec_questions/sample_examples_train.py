import pandas as pd
df = pd.read_csv('Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_questions/data/trec_train.csv')

df_selected_columns = df[['text', 'coarse_label']]

sampled_df = df_selected_columns.groupby('coarse_label').apply(lambda x: x.sample(n=5, random_state=42)).reset_index(drop=True) # Samples 5 examples for each Category

sampled_df.to_csv('Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_questions/data/sampled_data_5.csv', index=False)
print(sampled_df)