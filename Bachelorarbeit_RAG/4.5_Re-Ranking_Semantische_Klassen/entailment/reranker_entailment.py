import pandas as pd
import json
from classification_functions import classify_entailment

filename = "Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/entailment/data/pubmed/result_hybrid_pubmed_scores.pkl"
df = pd.read_pickle(filename)

df = df[df['matched'] == True]

all_rows_json_data = []

for index, row in df.iterrows():
    print("Processing index: " + str(index))
    ids = row['retrieved_hash_pos_in_doc']
    passages = row['retrieved_passages']
    question = row['question']
    true_pos = row['true_doc_hash_pos_in_doc']

    passages_list = [
        {
            'retrieved_hash_pos_in_doc': id_,
            'passage_text': passage,
            'old_position_in_list': idx + 1,
            'entailment_result': classify_entailment(question, passage),
            'correct_passage': id_ == true_pos
        }
        for idx, (id_, passage) in enumerate(zip(ids, passages))
    ]

    sorted_passages = sorted(passages_list, key=lambda x: x['entailment_result'] != 'entailment') # sorted() is stable-sorting

    for new_idx, item in enumerate(sorted_passages):
        item['new_position_in_list'] = new_idx + 1

    new_true_position = next((item['new_position_in_list'] for item in sorted_passages if item['correct_passage']), None)

    df.at[index, 'new_true_passage_position'] = new_true_position

    row_json = {
        'index': index,
        'question': question,
        'new_true_passage_position': new_true_position,
        'passages': sorted_passages
    }
    all_rows_json_data.append(row_json)

json_output = json.dumps(all_rows_json_data, indent=4)
with open('Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/entailment/data/pubmed/classification_archive/results_pubmed_k10_entailment.json', 'w', encoding='utf-8') as f:
    f.write(json_output)

print(df)
df.to_pickle('Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/entailment/data/pubmed/reranked_result_pubmed_hybrid_10.pkl')
df.to_csv('Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/entailment/data/pubmed/reranked_result_pubmed_hybrid_10.csv', index=False)