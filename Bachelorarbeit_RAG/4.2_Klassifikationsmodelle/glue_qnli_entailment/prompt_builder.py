import pandas as pd

df = pd.read_csv('Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/glue_qnli_entailment/data/sampled_data_10.csv')

prompt_header = """
Classify this question and sentence pair based on its entailment in one of these categories: entailment, not_entailment.

The label 'entailment' when the sentence contains the answer to the question.
The label is labeled 'not_entailment' when the sentence does not contain the answer to the question.

Here are some example questions together with the class label.
"""

few_shot_examples = []
for index, row in df.iterrows():
    example = f"""
    Question: {row['question']}
    Sentence: {row['sentence']}
    Label: {row['label']}
    """
    few_shot_examples.append(example)

full_prompt = prompt_header + "\n".join(few_shot_examples)

print(full_prompt)
with open('Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/glue_qnli_entailment/entailment_prompt.txt', 'w', encoding='utf-8') as file:
    file.write(full_prompt)