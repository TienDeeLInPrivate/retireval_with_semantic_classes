from datasets import load_dataset

dataset = load_dataset("glue", "qnli")

validation_dataset = dataset["validation"]
df_QNLI_validation = validation_dataset.to_pandas()

train_dataset = dataset["train"]
df_QNLI_train = train_dataset.to_pandas()

label_mapping = {
    0: 'entailment',
    1: 'not_entailment'
}

df_QNLI_validation['label'] = df_QNLI_validation['label'].replace(label_mapping)
df_QNLI_train['label'] = df_QNLI_train['label'].replace(label_mapping)

#print(df_QNLI_validation)
#print(df_QNLI_train)

df_QNLI_validation.to_csv("Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/glue_qnli_entailment/data/qnli_validation.csv", index=False)
df_QNLI_train.to_csv("Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/glue_qnli_entailment/data/qnli_train.csv", index=False)
