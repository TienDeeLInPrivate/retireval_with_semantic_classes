from dotenv import load_dotenv
import pandas as pd
from genai import Credentials, Client
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions
from sklearn.metrics import accuracy_score

load_dotenv()
credentials = Credentials.from_env()
client = Client(credentials=credentials)

df_TREC_test = pd.read_csv("Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_questions/data/trec_test.csv")

for index, row in df_TREC_test.iterrows():
    print(index)
    response = list(
        client.text.generation.create(
            model_id="google/flan-ul2",
            inputs=[f"""Classify this question based on its intent in one of these categories: abbreviation, entity, description, human, location, or numeric. 
                    Focus on the interrogative pronouns.
                    The result of a query can only consist of a single word. In Example: description
                    
                    Here are some example questions together with the class label of the question:
                                
                    Question: What is Mikhail Gorbachev 's middle initial ?
                    Label: abbreviation
                    
                    Question: What is the full form of .com ?
                    Label: abbreviation
                    
                    Question: What is LMDS ?
                    Label: abbreviation
                    
                    Question: What does e.g. stand for ?
                    Label: abbreviation
                    
                    Question: When reading classified ads , what does EENTY : other stand for ?
                    Label: abbreviation
                    
                    Question: What is a handheld PC ?
                    Label: description
                    
                    Question: What does the name Billie mean ?
                    Label: description
                    
                    Question: How is Answers.com funded ?
                    Label: description
                    
                    Question: What is troilism ?
                    Label: description
                    
                    Question: What is the purpose of BIOS ?
                    Label: description
                    
                    Question: What is the term for the side of the mountain that faces the prevailing winds ?
                    Label: entity
                    
                    Question: What 's played at Wembley Stadium , London , every May ?
                    Label: entity
                    
                    Question: What is a fear of color ?
                    Label: entity
                    
                    Question: What future movie treat was introduced to American colonists in 1603 by Native Americans ?
                    Label: entity
                    
                    Question: What 's the official language of Algeria ?
                    Label: entity
                    
                    Question: Who gave Abbie Hoffman his first dose of LSD ?
                    Label: human
                    
                    Question: What singer became despondent over the death of Freddie Prinze , quit show business , and then quit the business ?
                    Label: human
                    
                    Question: What 19th-century writer had a country estate on the Hudson dubbed Sunnyside ?
                    Label: human
                    
                    Question: Name the three races unleashed by the Celestials in Marvel comics .
                    Label: human
                    
                    Question: What actor and actress have made the most movies ?
                    Label: human
                    
                    Question: Where can I get piano music for the Jamiroquai song Everyday for the midi ?
                    Label: location
                    
                    Question: What state is known as the Hawkeye State ?
                    Label: location
                    
                    Question: Where does Mother Angelica live ?
                    Label: location
                    
                    Question: Where is Erykah Badu originally from ?
                    Label: location
                    
                    Question: Where did guinea pigs originate ?
                    Label: location
                    
                    Question: How many gallons of water go over Niagra Falls every second ?
                    Label: numeric
                    
                    Question: What time of year do most people fly ?
                    Label: numeric
                    
                    Question: What is the weight of a teaspoon of matter in a black hole ?
                    Label: numeric
                    
                    Question: How many colors are there in a rainbow ?
                    Label: numeric
                    
                    Question: When did Mount St. Helen last have a significant eruption ?
                    Label: numeric

                    Question: {row['text']}
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
    df_TREC_test.loc[index, 'output_ul2'] = result.generated_text

    print(result.generated_text)

accuracy_sklearn = accuracy_score(df_TREC_test['coarse_label'], df_TREC_test['output_ul2'])
print(df_TREC_test)
df_TREC_test.to_csv("Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/trec_questions/data/trec_test_results_ul2.csv", index=False)
print("Accuracy: ")
print(accuracy_sklearn)
# Accuracy: 0.97