import pandas as pd
import json

filename = "Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/wikipedia/result_hybrid_wiki_ibm_scores.pkl"
df = pd.read_pickle(filename)

with open('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/wikipedia/classification_archive/classification_results.json', 'r') as file:
    trec_data = json.load(file)

def get_trec_result(trec_data, question, hash_id):
    for entry in trec_data:
        if entry['question'] == question:
            for passage in entry['passages']:
                if passage['retrieved_hash_pos_in_doc'] == hash_id:
                    return passage['passage_question_match']
    print("None found ERROR")
    return None  
df = df[df['matched'] == True]

all_rows_json_data = []

weight = 0.24 # Set the best weight


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
            'trec_passage_match_result': (passage_question_match := get_trec_result(trec_data, question, id_)),
            'correct_passage': id_ == true_pos,
            'old_retrieval_score': retrieval_score,
            'new_retrieval_score': retrieval_score + (weight * (1 if passage_question_match == True else 0))
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
with open('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/wikipedia/classification_archive/best_weight_results.json', 'w', encoding='utf-8') as f:
    f.write(json_output)

print(df)
df.to_pickle('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/wikipedia/integrated_retrieval_result_wikipedia_best_weight.pkl')
df.to_csv('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/wikipedia/integrated_retrieval_result_wikipedia_best_weight.csv', index=False)
