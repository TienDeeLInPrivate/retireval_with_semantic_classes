import pandas as pd
import json

filename = "Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/entailment/data/pubmed/result_hybrid_pubmed_scores.pkl"
df = pd.read_pickle(filename)

weights = []
mrr_scores = []

with open('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/entailment/data/pubmed/classification_archive/results_pubmed_k10_entailment.json', 'r') as file:
    entailment_data = json.load(file)

def get_entailment_result(entailment_data, question, hash_id):
    for entry in entailment_data:
        if entry['question'] == question:
            for passage in entry['passages']:
                if passage['retrieved_hash_pos_in_doc'] == hash_id:
                    return passage['entailment_result']
    print("None found ERROR")
    return None  # Return None if no match is found, should never happen

df = df[df['matched'] == True]

all_rows_json_data = []

mrr_old = 0
weight = 0
best_weight = -1
for i in range (0, 1000):
    print("Current Weight:")
    print(weight)
    print("Best Weight:")
    print(best_weight)
    weight = weight + 0.01
    weights.append(weight)
    for index, row in df.iterrows():
        #print("Processing index: " + str(index))
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
                'entailment_result': (entailment := get_entailment_result(entailment_data, question, id_)),
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


    def calculate_mrr(df, position_column):
        df['reciprocal_rank'] = 1 / df[position_column]
        return df['reciprocal_rank'].mean()

    mrr_new = calculate_mrr(df, 'new_true_passage_position')
    mrr_scores.append(mrr_new)
    if mrr_new > mrr_old:
        mrr_old = mrr_new
        best_weight = weight

print(best_weight)
print(mrr_old) # Needs to print mrr_old because the best mrr will be saved in here after switch

df_best_weight_search = pd.DataFrame({
    'Weight': weights,
    'MRR': mrr_scores
})
df_best_weight_search.to_csv("Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/entailment/data/pubmed/weight_search.csv")

# Wikipedia:
# Current best weight: 1.29
# Best MRR: 0.9007

# PubMed:
# Current best weight: 1.480000000000001
# Best MRR: 0.8769778834377926