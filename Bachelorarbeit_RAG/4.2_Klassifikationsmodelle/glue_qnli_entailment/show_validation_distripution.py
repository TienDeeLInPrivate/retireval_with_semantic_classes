import pandas as pd
import plotly.express as px

df = pd.read_csv('Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/glue_qnli_entailment/data/qnli_validation.csv')  

count = df['label'].value_counts().reset_index()
count.columns = ['label', 'count']
count['percentage'] = 100 * count['count'] / count['count'].sum()

fig = px.bar(count, x='label', y='count', text='percentage',
             labels={'count': 'Anzahl der Daten', 'label': 'Label'},
             title='Verteilung der QNLI Testdaten')

fig.update_layout(
    title={
        'text': 'Verteilung der QNLI Testdaten',
        'y': 0.94,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    title_font_size=20,
    font=dict(
        family="Arial, sans-serif",
        size=16,
    )
)
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_traces(width=0.5)
fig.show()