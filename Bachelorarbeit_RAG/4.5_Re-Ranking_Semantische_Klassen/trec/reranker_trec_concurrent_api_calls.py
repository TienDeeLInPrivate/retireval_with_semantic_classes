import pandas as pd
import numpy as np
from multiprocessing import Pool
from classification_trec import classify_passage_trec, classify_question_trec
import json

def process_chunk(chunk, df_wikipedia_classified_passages):
    all_rows_json_data = []
    for index, row in chunk.iterrows():
        print(f"Processing index: {index}")
        ids = row['retrieved_hash_pos_in_doc']
        passages = row['retrieved_passages']
        question = row['question']
        true_pos = row['true_doc_hash_pos_in_doc']
        question_classification = classify_question_trec(question)

        passages_list = [
            {
                'retrieved_hash_pos_in_doc': id_,
                'passage_text': passage,
                'old_position_in_list': idx + 1,
                'question_classification_result': question_classification,
                'passage_classification_result': (passage_classifications := classify_passage_trec(passage, df_wikipedia_classified_passages)),
                'passage_question_match': question_classification in passage_classifications
            }
            for idx, (id_, passage) in enumerate(zip(ids, passages))
        ]

        sorted_passages = sorted(passages_list, key=lambda x: not x['passage_question_match'])

        new_true_position = next((new_idx + 1 for new_idx, item in enumerate(sorted_passages) if item['retrieved_hash_pos_in_doc'] == true_pos), None)
        chunk.at[index, 'new_true_passage_position'] = new_true_position

        all_rows_json_data.append({
            'index': index,
            'question': question,
            'new_true_passage_position': new_true_position,
            'passages': sorted_passages
        })

    return chunk, all_rows_json_data

def main():
    filename = "Bachelorarbeit_RAG/4.3_Re-Ranking_Semantische_Klassen/trec/data/pubmed/result_hybrid_pubmed_scores.pkl"
    df = pd.read_pickle(filename)
    df_classified_passages = pd.read_pickle("Bachelorarbeit_RAG/4.3_Re-Ranking_Semantische_Klassen/trec/data/wikipedia/wiki_ibm_qa_pairs_mistral_passages_classified.pkl")

    df = df[df['matched'] == True]

    # 100 chunks possible after bam api concurrency limit raised (request sent to BAM product management)
    chunks = np.array_split(df, 100)

    with Pool(100) as pool:
        results = pool.starmap(process_chunk, [(chunk, df_classified_passages) for chunk in chunks])

    updated_df = pd.concat([res[0] for res in results], ignore_index=True)
    all_data_json = [item for sublist in [res[1] for res in results] for item in sublist]

    # Saving to Dataframe and JSON very important, results can be reused
    updated_df.to_pickle('Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/trec/data/pubmed/reranked_result_pubmed_hybrid_10.pkl')
    updated_df.to_csv('Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/trec/data/pubmed/reranked_result_pubmed_hybrid_10.csv', index=False)
    with open('Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/trec/data/pubmed/classification_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_data_json, f, indent=4)

    print("Data processing complete.")

if __name__ == "__main__":
    main()