import pandas as pd
import plotly.express as px

filename_pubmed_integrated = "Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/pubmed/integrated_retrieval_result_pubmed_best_weight.pkl"
filename_pubmed = "Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/pubmed/reranked_result_pubmed_hybrid_10.pkl"

df_pubmed = pd.read_pickle(filename_pubmed)
df_pubmed_integrated = pd.read_pickle(filename_pubmed_integrated)

max_k = 10
def top_k_accuracy(df, column):
    accuracies = []
    for k in range(1, max_k + 1):
        accuracy = (df[column] <= k).mean() * 100
        accuracies.append(accuracy)
    return accuracies

new_accuracies_pubmed_integrated = top_k_accuracy(df_pubmed_integrated, 'new_true_passage_position')
old_accuracies_pubmed = top_k_accuracy(df_pubmed, 'old_true_passage_position')
new_accuracies_pubmed = top_k_accuracy(df_pubmed, 'new_true_passage_position')

top_k_df = pd.DataFrame({
    'Top K': range(1, 11),
    'Originale Abrufung (PubMed)': old_accuracies_pubmed,
    'Abrufung mit Re-Ranking (PubMed)': new_accuracies_pubmed,
    'Integrierte Abrufung (Pubmed)': new_accuracies_pubmed_integrated
})

def calculate_mrr(df, position_column):
    df['reciprocal_rank'] = 1 / df[position_column]
    return df['reciprocal_rank'].mean()


mrr_old_pubmed = calculate_mrr(df_pubmed, 'old_true_passage_position')
mrr_new_pubmed = calculate_mrr(df_pubmed, 'new_true_passage_position')
mrr_new_pubmed_integrated = calculate_mrr(df_pubmed_integrated, 'new_true_passage_position')

top_k_df_transposed = top_k_df.set_index('Top K').transpose()
mrr_values = {
    'Originale Abrufung (PubMed)': mrr_old_pubmed,
    'Abrufung mit Re-Ranking (PubMed)': mrr_new_pubmed,
    'Integrierte Abrufung (Pubmed)': mrr_new_pubmed_integrated
}
top_k_df_transposed['MRR'] = pd.Series(mrr_values)  

top_k_df_transposed.iloc[:, :-1] = top_k_df_transposed.iloc[:, :-1].round(2)  
top_k_df_transposed['MRR'] = top_k_df_transposed['MRR'].round(4)  

print(top_k_df_transposed)
#top_k_df_transposed.to_clipboard()

fig = px.line(top_k_df, x='Top K', y=top_k_df.columns[1:], markers=True,
              labels={"value": "Genauigkeit", "variable": "Methode"},
              title="Vergleich der Top-K Genauigkeit: Integrierte Abrufung (TREC) vs Re-Ranking vs Original")
fig.update_layout(
    xaxis=dict(tickmode='linear', tick0=1, dtick=1, title_font=dict(size=25), tickfont=dict(size=21)),
    yaxis=dict(title_font=dict(size=25), tickfont=dict(size=21)),
    title=dict(text="Vergleich der Top-K Genauigkeit: Integrierte Abrufung (TREC) vs Re-Ranking vs Original", x=0.5, xanchor='center', font=dict(size=29)),
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
colors = ['red', 'blue', 'green']  
for trace, color in zip(fig.data, colors):
    trace.line.color = color

fig.show()