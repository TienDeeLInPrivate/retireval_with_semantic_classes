from dotenv import load_dotenv
import pandas as pd
from genai import Credentials, Client
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions

load_dotenv()
credentials = Credentials.from_env()
client = Client(credentials=credentials)

pubmed_file = "Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/qa_mistral_pubmed.csv"
wiki_ibm_file = "Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/wiki_ibm_qa_pairs_mistral.csv"

df = pd.read_csv(pubmed_file)
df = df[['context']]
df = df.drop_duplicates()

files = ["Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/prompts/abbreviation_prompt.txt",
         "Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/prompts/description_prompt.txt",
         "Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/prompts/entity_prompt.txt",
         "Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/prompts/human_prompt.txt",
         "Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/prompts/location_prompt.txt",
         "Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/prompts/numeric_prompt.txt"]


prompt_types = ["abbreviation", "description", "entity", "human", "location", "numeric"]
prompts = {}

for file_path, passage_class in zip(files, prompt_types):
    with open(file_path, 'r') as file:
        file_contents = file.read()
        prompts[passage_class] = file_contents


#print(prompts)

for passage_class, prompt_text in prompts.items():
    for index, row in df.iterrows():
        print(index)
        print("Class: " + passage_class)
        try:
            response = list(
                client.text.generation.create(
                    model_id="google/flan-ul2",
                    inputs=[prompt_text + f""" 
                            
                            Target Text-Passage: {row['context']}
                            Label:
                            """],
                    parameters=TextGenerationParameters(
                        temperature=0,
                        max_new_tokens=50,
                        return_options=TextGenerationReturnOptions(input_text=True),
                    ),
                )
            )
            result = response[0].results[0]
            df.loc[index, passage_class] = result.generated_text
            print("Result_label: " + result.generated_text)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            df.loc[index, passage_class] = "Error"  
print(df)
df.to_pickle("Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_passages/pubmed_qa_pairs_mistral_passages_classified.pkl")