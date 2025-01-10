import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency

# Load the dataset
with open('./data/converted_segmental_information_with_speech_analysis.json', 'r') as f:
    data = json.load(f)

    data = [ x for x in data.values() if x['speech_analysis'] != 'EMPTY' and x['speech_analysis'] != 'ERROR' ]

# Convert JSON data to DataFrame
records = []
for value in data:
    records.append({
        'utterance_id': value['utterance_id'],
        'interlocutor': value['interlocutor'],
        'client_talk_type': value['client_talk_type'],
        'speech_analysis': value['speech_analysis'],
        'main_therapist_behaviour': value['main_therapist_behaviour']
    })

df = pd.DataFrame(records)

# Encode categorical variables
speech_encoder = df['speech_analysis'].astype('category').cat.codes
talk_encoder = df['client_talk_type'].astype('category').cat.codes
tb_encoder = df['main_therapist_behaviour'].astype('category').cat.codes

df['speech_analysis_encoded'] = speech_encoder
df['client_talk_type_encoded'] = talk_encoder
df['main_therapist_behaviour_encoded'] = tb_encoder

# Split into client and therapist utterances
client_df = df[df['interlocutor'] == 'client'].reset_index(drop=True)
therapist_df = df[df['interlocutor'] == 'therapist'].reset_index(drop=True)

# Plot distribution of speech_analysis across client_talk_type
plt.figure(figsize=(10, 6))
# client_counts = pd.crosstab(client_df['client_talk_type'], client_df['speech_analysis'])
# client_counts.plot(kind='bar', stacked=True)
# plt.xlabel("Client Talk Type")
# plt.ylabel("Count of Speech Analysis")
# plt.title("Distribution of Speech Analysis Across Client Talk Types")
# plt.legend(title="Speech Analysis")
# plt.tight_layout()
# plt.savefig("./fig.png")
therapist_counts = pd.crosstab(therapist_df['main_therapist_behaviour'], therapist_df['speech_analysis'])
therapist_counts.plot(kind='bar', stacked=True)
plt.xlabel("Main Therapist Behaviour")
plt.ylabel("Count of Speech Analysis")
plt.title("Distribution of Speech Analysis Across Main Therapist Behaviours")
plt.legend(title="Speech Analysis")
plt.tight_layout()
plt.savefig("./fig.png")

