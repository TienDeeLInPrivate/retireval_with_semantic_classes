import pandas as pd
import json

filename = "Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/pubmed/result_hybrid_pubmed_scores.pkl"
df = pd.read_pickle(filename)

with open('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/pubmed/classification_archive/classification_results.json', 'r') as file:
    entailment_data = json.load(file)

def get_trec_result(entailment_data, question, hash_id):
    for entry in entailment_data:
        if entry['question'] == question:
            for passage in entry['passages']:
                if passage['retrieved_hash_pos_in_doc'] == hash_id:
                    return passage['passage_question_match']
    print("None found ERROR")
    return None  

df = df[df['matched'] == True]

all_rows_json_data = []

weight = 0.16 # Set the best weight


for index, row in df.iterrows():
    print("Processing index: " + str(index))
    ids = row['retrieved_hash_pos_in_doc']
    passages = row['retrieved_passages']
    question = row['question']
    true_pos = row['true_doc_hash_pos_in_doc']
    retrieved_scores = row['retrieved_scores']

    passages_list = [
        {
            'retrieved_hash_pos_in_doc': id_,
            'passage_text': passage,
            'old_position_in_list': idx + 1,
            'entailment_result': (entailment := get_trec_result(entailment_data, question, id_)),
            'correct_passage': id_ == true_pos,
            'old_retrieval_score': retrieval_score,
            'new_retrieval_score': retrieval_score + (weight * (1 if entailment == "entailment" else 0))
        }
        for idx, (id_, passage, retrieval_score) in enumerate(zip(ids, passages, retrieved_scores))
    ]

    sorted_passages = sorted(passages_list, key=lambda x: x['new_retrieval_score'], reverse=True)

    for new_idx, item in enumerate(sorted_passages):
        item['new_position_in_list'] = new_idx + 1

    new_true_position = next((item['new_position_in_list'] for item in sorted_passages if item['correct_passage']), None)

    df.at[index, 'new_true_passage_position'] = new_true_position

    row_json = {
        'index': index,
        'question': question,
        'new_true_passage_position': new_true_position,
        'passages': passages_list
    }
    all_rows_json_data.append(row_json)

json_output = json.dumps(all_rows_json_data, indent=4)
with open('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/pubmed/classification_archive/best_weight_results.json', 'w', encoding='utf-8') as f:
    f.write(json_output)

print(df)
df.to_pickle('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/pubmed/integrated_retrieval_result_pubmed_best_weight.pkl')
df.to_csv('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/pubmed/integrated_retrieval_result_pubmed_best_weight.csv', index=False)
