from dotenv import load_dotenv
import pandas as pd
from genai import Credentials, Client
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions
from sklearn.metrics import accuracy_score

load_dotenv()
credentials = Credentials.from_env()
client = Client(credentials=credentials)

df_QNLI_test = pd.read_csv("Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/glue_qnli_entailment/data/qnli_validation.csv")

for index, row in df_QNLI_test.iterrows():
    print(index)
    response = list(
        client.text.generation.create(
            model_id="google/flan-ul2",
            inputs=[f"""Classify this question and sentence pair based on its entailment in one of these categories: entailment, not_entailment.

                        The label 'entailment' when the sentence contains the answer to the question.
                        The label is labeled 'not_entailment' when the sentence does not contain the answer to the question.

                        Here are some example questions together with the class label:
                            
                            Question: Who did NASA recruit by using flawed safety numbers?
                            Sentence: He concluded that the space shuttle reliability estimate by NASA management was fantastically unrealistic, and he was particularly angered that NASA used these figures to recruit Christa McAuliffe into the Teacher-in-Space program.
                            Label: entailment
                            
                            Question: How much solar energy is captured by photosynthesis?
                            Sentence: Photosynthesis captures approximately 3,000 EJ per year in biomass.
                            Label: entailment
                            
                            Question: What does the CAR get help with with regards to communication from ITU-D?
                            Sentence: In addition, the Central African Republic receives international support on telecommunication related operations from ITU Telecommunication Development Sector (ITU-D) within the International Telecommunication Union to improve infrastructure.
                            Label: entailment
                            
                            Question: On Indian Independence Day, kites are flown by citizens which symbolize what concept?
                            Sentence: Most Delhiites celebrate the day by flying kites, which are considered a symbol of freedom.
                            Label: entailment
                            
                            Question: What types of Christianity do Quakers belong to?
                            Sentence: They include those with evangelical, holiness, liberal, and traditional conservative Quaker understandings of Christianity.
                            Label: entailment
                            
                            Question: The Further and Higher Education Act 1992 allows polytechnics to award degrees without what organization's approval?
                            Sentence: This meant that Polytechnics could confer degrees without the oversight of the national CNAA organization.
                            Label: entailment
                            
                            Question: In what square is the theater named after Lee Strasberg located?
                            Sentence: The Lee Strasberg Theatre and Film Institute is in Union Square, and Tisch School of the Arts is based at New York University, while Central Park SummerStage presents performances of free plays and music in Central Park.
                            Label: entailment
                            
                            Question: Who are the famous Venezuelen mandolinist?
                            Sentence: Today, Venezuelan mandolists include an important group of virtuoso players and ensembles such as Alberto Valderrama, Jesus Rengel, Ricardo Sandoval, Saul Vera, and Cristobal Soto.
                            Label: entailment
                            
                            Question: What take place during SWS?
                            Sentence: System consolidation takes place during slow-wave sleep (SWS).
                            Label: entailment
                            
                            Question: Rayon comes from what plant product?
                            Sentence: Products made from cellulose include rayon and cellophane, wallpaper paste, biobutanol and gun cotton.
                            Label: entailment
                            
                            Question: How is it postulated that Mars life might have evolved?
                            Sentence: Those features can also be observed in algae and cyanobacteria, suggesting that these are adaptations to the conditions prevailing in Antarctica.
                            Label: not_entailment

                            Question: What determines how deep a tester will go during regression?
                            Sentence: They can either be complete, for changes added late in the release or deemed to be risky, or be very shallow, consisting of positive tests on each feature, if the changes are early in the release or deemed to be of low risk.
                            Label: not_entailment

                            Question: What was the job title of Ed Policy?
                            Sentence: Progress on the return stalled, and no announcements were made regarding the future of the league.
                            Label: not_entailment

                            Question: What did Ibn Sina receive as payment for helping the emir?
                            Sentence: Ibn Sina's first appointment was that of physician to the emir, Nuh II, who owed him his recovery from a dangerous illness (997).
                            Label: not_entailment

                            Question: How much did Bell et al. try to sell his patent for?
                            Sentence: By then, the Bell company no longer wanted to sell the patent.
                            Label: not_entailment

                            Question: What is Spielberg's most common theme?
                            Sentence: The notable absence of Elliott's father in E.T., is the most famous example of this theme.
                            Label: not_entailment
                            
                            Question: What does the oldest know term for Egypt translate to?
                            Sentence: The name is of Semitic origin, directly cognate with other Semitic words for Egypt such as the Hebrew מִצְרַיִם (Mitzráyim).
                            Label: not_entailment

                            Question: Westminster Abbey was the third highest place of learning after which two places?
                            Sentence: It was here that the first third of the King James Bible Old Testament and the last half of the New Testament were translated.
                            Label: not_entailment

                            Question: What famous encyclopedia contains a Russian back-transliteration of Estonian?
                            Sentence: It should be noted that Estonian words and names quoted in international publications from Soviet sources are often back-transliterations from the Russian transliteration.
                            Label: not_entailment

                            Question: When was The Sound Pattern of English published?
                            Sentence: An important consequence of the influence SPE had on phonological theory was the downplaying of the syllable and the emphasis on segments.
                            Label: not_entailment

                            Question: {row['question']}
                            Sentence: {row['sentence']}
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
    df_QNLI_test.loc[index, 'output_ul2'] = result.generated_text

    print(result.generated_text)

accuracy_sklearn = accuracy_score(df_QNLI_test['label'], df_QNLI_test['output_ul2'])
print(df_QNLI_test)
df_QNLI_test.to_csv("Bachelorarbeit_RAG/4.2_Klassifikationsmodelle/glue_qnli_entailment/data/qnli_test_results_ul2.csv", index=False)
print("Accuracy: ")
print(accuracy_sklearn)

# Accuracy_old: 0.9487461101958631
# Accuracy_current: 0.942339373970346