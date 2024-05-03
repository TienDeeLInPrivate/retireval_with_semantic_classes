import pandas as pd
import plotly.express as px

results_df = pd.read_csv('Bachelorarbeit_RAG/4.6_Integrierte_Abrufung/trec/data/wikipedia/weight_search.csv')

fig = px.line(
    results_df,
    x='Weight',
    y='MRR',
    title='MRR in Abhängigkeit von der Klassifizierungsgewichtung (TREC-Klassen, Wikipedia)',
    labels={'Weight': 'Gewichtung', 'MRR': 'MRR'}
)

fig.update_traces(mode='lines+markers')

fig.update_layout(
    xaxis_title='Weight',
    yaxis_title='MRR',
    xaxis=dict(showgrid=True, title_font=dict(size=24), tickfont=dict(size=22)),
    yaxis=dict(showgrid=True, title_font=dict(size=24), tickfont=dict(size=22)),
    plot_bgcolor='white',
    title={
        'text': 'MRR in Abhängigkeit von der Klassifizierungsgewichtung (TREC-Klassen, Wikipedia)',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    title_font=dict(size=26)
)

fig.show()