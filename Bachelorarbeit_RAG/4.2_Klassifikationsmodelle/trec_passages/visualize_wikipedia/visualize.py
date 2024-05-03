import pandas as pd
import plotly.express as px

file = "Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/wiki_ibm_qa_pairs_mistral_passages_classified.pkl"
df = pd.read_pickle(file)

counts = {
    'abbreviation': (df['abbreviation'] == 'abbreviation').sum(),
    'numeric': (df['numeric'] == 'numeric').sum(),
    'location': (df['location'] == 'location').sum(),
    'description': (df['description'] == 'description').sum(),
    'entity': (df['entity'] == 'entity').sum(),
    'human': (df['human'] == 'human').sum(),
    'error': df.apply(lambda row: 'Error' in row.values, axis=1).sum()
}

#print(counts)
counts_df = pd.DataFrame(list(counts.items()), columns=['Label', 'Count'])
counts_df['Percentage'] = (counts_df['Count'] / len(df)) * 100

#print(counts_df)

order = ['description', 'numeric', 'entity', 'location', 'human', 'abbreviation', 'error']

fig = px.bar(counts_df, x='Label', y='Percentage',
             text='Percentage',
             category_orders={'Label': order},
             labels={'Percentage': 'Anteil der Passagen'},
             title='Prozentuale Verteilung der Passagen auf Informationskategorien')
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

fig.update_layout(
    yaxis=dict(range=[0, 100]),
    title=dict(text='Prozentuale Verteilung der Passagen auf Informationskategorien', x=0.5),
    font=dict(size=16, family="Arial, sans-serif") 
)

fig.show()