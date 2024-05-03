import pandas as pd

from datasets import load_dataset
dataset = load_dataset("trec")

train_dataset = dataset["train"]
df_TREC_train = train_dataset.to_pandas()

test_dataset = dataset["test"]
df_TREC_test = test_dataset.to_pandas()

df_TREC_test = df_TREC_test[['text', 'coarse_label']]
df_TREC_train = df_TREC_train[['text', 'coarse_label']]

label_mapping = {
    0: 'abbreviation',
    1: 'entity',
    2: 'description',
    3: 'human',
    4: 'location',
    5: 'numeric'
}

df_TREC_test['coarse_label'] = df_TREC_test['coarse_label'].replace(label_mapping)
df_TREC_train['coarse_label'] = df_TREC_train['coarse_label'].replace(label_mapping)

print(df_TREC_train)
print(df_TREC_test)

df_TREC_test.to_csv("Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_questions/data/trec_test.csv", index=False)
df_TREC_train.to_csv("Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_questions/data/trec_train.csv", index=False)

