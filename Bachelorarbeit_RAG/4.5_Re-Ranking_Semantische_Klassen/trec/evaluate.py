import pandas as pd
import plotly.express as px

filename_wikipedia = "Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/trec/data/wikipedia/reranked_result_wikipedia_hybrid_10.pkl"
filename_pubmed = "Bachelorarbeit_RAG/4.5_Re-Ranking_Semantische_Klassen/trec/data/pubmed/reranked_result_pubmed_hybrid_10.pkl"
df_wikipedia = pd.read_pickle(filename_wikipedia)
df_pubmed = pd.read_pickle(filename_pubmed)

max_k = 10
def top_k_accuracy(df, column):
    accuracies = []
    for k in range(1, max_k + 1):
        accuracy = (df[column] <= k).mean() * 100
        accuracies.append(accuracy)
    return accuracies

old_accuracies_wikipedia = top_k_accuracy(df_wikipedia, 'old_true_passage_position')
new_accuracies_wikipedia = top_k_accuracy(df_wikipedia, 'new_true_passage_position')
old_accuracies_pubmed = top_k_accuracy(df_pubmed, 'old_true_passage_position')
new_accuracies_pubmed = top_k_accuracy(df_pubmed, 'new_true_passage_position')

top_k_df = pd.DataFrame({
    'Top K': range(1, 11),
    'Originale Abrufung (Wikipedia)': old_accuracies_wikipedia,
    'Abrufung mit Re-Ranking (Wikipedia)': new_accuracies_wikipedia,
    'Originale Abrufung (PubMed)': old_accuracies_pubmed,
    'Abrufung mit Re-Ranking (PubMed)': new_accuracies_pubmed
})

def calculate_mrr(df, position_column):
    df['reciprocal_rank'] = 1 / df[position_column]
    return df['reciprocal_rank'].mean()

mrr_old_wikipedia = calculate_mrr(df_wikipedia, 'old_true_passage_position')
mrr_new_wikipedia = calculate_mrr(df_wikipedia, 'new_true_passage_position')
mrr_old_pubmed = calculate_mrr(df_pubmed, 'old_true_passage_position')
mrr_new_pubmed = calculate_mrr(df_pubmed, 'new_true_passage_position')

top_k_df_transposed = top_k_df.set_index('Top K').transpose()
mrr_values = {
    'Originale Abrufung (Wikipedia)': mrr_old_wikipedia,
    'Abrufung mit Re-Ranking (Wikipedia)': mrr_new_wikipedia,
    'Originale Abrufung (PubMed)': mrr_old_pubmed,
    'Abrufung mit Re-Ranking (PubMed)': mrr_new_pubmed
}
top_k_df_transposed['MRR'] = pd.Series(mrr_values)  

top_k_df_transposed.iloc[:, :-1] = top_k_df_transposed.iloc[:, :-1].round(2)  
top_k_df_transposed['MRR'] = top_k_df_transposed['MRR'].round(4)  

print(top_k_df_transposed)
top_k_df_transposed.to_clipboard()

fig = px.line(top_k_df, x='Top K', y=top_k_df.columns[1:], markers=True,
              labels={"value": "Genauigkeit", "variable": "Methode"},
              title="Vergleich der Top-K Genauigkeit: Re-Ranking (TREC) vs Original")
fig.update_layout(
    xaxis=dict(tickmode='linear', tick0=1, dtick=1, title_font=dict(size=25), tickfont=dict(size=21)),
    yaxis=dict(title_font=dict(size=25), tickfont=dict(size=21)),
    title=dict(text="Vergleich der Top-K Genauigkeit: Re-Ranking (TREC) vs Original", x=0.5, xanchor='center', font=dict(size=29)),
    legend_title_text='Methode',
     legend=dict(
        title_font_size=24,
        font=dict(size=22),
        x=0.968, 
        y=0.05,  
        xanchor='right',  
        yanchor='bottom'  
    )
)
colors = ['#f7a6a6', '#d62728', '#7fb0e1', '#1f77b4']  
for trace, color in zip(fig.data, colors):
    trace.line.color = color

#fig.show()

