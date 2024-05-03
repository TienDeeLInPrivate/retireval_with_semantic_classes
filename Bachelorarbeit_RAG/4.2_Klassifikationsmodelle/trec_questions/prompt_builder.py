import pandas as pd

df = pd.read_csv('Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_questions/data/sampled_data_5.csv')

prompt_header = """
                    Classify this question based on its intent in one of these categories: abbreviation, entity, description, human, location, or numeric. 
                    Focus on the interrogative pronouns.
                    The result of a query can only consist of a single word. In Example: description
                    
                    Here are some example questions together with the class label of the question:
                """

few_shot_examples = []
for index, row in df.iterrows():
    example = f"""
    Question: {row['text']}
    Label: {row['coarse_label']}
    """
    few_shot_examples.append(example)

complete_prompt = prompt_header + "\n".join(few_shot_examples)

print(complete_prompt)
with open('Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_questions/trec_question_intent_prompt.txt', 'w', encoding='utf-8') as file:
    file.write(complete_prompt)