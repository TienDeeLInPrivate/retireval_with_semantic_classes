import pandas as pd

df = pd.read_csv('Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/glue_qnli_entailment/data/qnli_train.csv')

df_selected_columns = df[['question', 'sentence', 'label']]

sampled_df = df_selected_columns.groupby('label').apply(lambda x: x.sample(n=10, random_state=42)).reset_index(drop=True)

sampled_df.to_csv('Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/glue_qnli_entailment/data/sampled_data_10.csv', index=False)
print(sampled_df)

