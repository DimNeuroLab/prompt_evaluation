import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 1. LOAD THE DATA FROM THE XLSX
# ---------------------------
#xlsx_file = "interactions.xlsx"  # Update with your actual file path
xlsx_file = "G:\\My Drive\\projects\\GPTInteractionAnnotationVonNeumidas\\ANNOTATIONS _FINAL_SOLMAZ.xlsx"#'von_neumidas_interactions.xlsx'

# Load the Excel file and get the list of sheets (each sheet represents one interaction)
xls = pd.ExcelFile(xlsx_file)
sheet_names = xls.sheet_names

# Read each sheet into a DataFrame and add a column with the sheet name (optional)
dfs = []
for sheet in sheet_names:
    df_sheet = pd.read_excel(xlsx_file, sheet_name=sheet)
    df_sheet['sheet'] = sheet  # Optional: track which sheet each row came from
    dfs.append(df_sheet)

# Concatenate all sheets into one DataFrame for analysis
data = pd.concat(dfs, ignore_index=True)

# Convert key columns to string (if needed)
data['pid'] = data['pid'].astype(str)
data['studyId'] = data['studyId'].astype(str)

print("Data head:")
print(data.head())
print("\nData info:")
print(data.info())

# ---------------------------
# 2. BASIC DESCRIPTIVE ANALYSES
# ---------------------------
# 2.1 Count the number of messages per study/session
session_message_counts = data.groupby('studyId')['text'].count().reset_index(name='message_count')
print("\nMessage counts per study/session:")
print(session_message_counts)

plt.figure(figsize=(8, 6))
sns.barplot(data=session_message_counts, x='studyId', y='message_count', palette='viridis')
plt.title('Number of Messages per Study/Session')
plt.xlabel('Study/Session')
plt.ylabel('Message Count')
plt.tight_layout()
plt.show()

# 2.2 Distribution of MIDAS speech acts across sessions
midas_distribution = data.groupby(['studyId', 'MIDAS'])['text'].count().reset_index(name='count')
print("\nMIDAS distribution per study/session:")
print(midas_distribution)

plt.figure(figsize=(12, 8))
sns.barplot(data=midas_distribution, x='studyId', y='count', hue='MIDAS')
plt.title('Distribution of MIDAS Speech Acts per Session')
plt.xlabel('Study/Session')
plt.ylabel('Count')
plt.legend(title='MIDAS Act', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---------------------------
# 3. ANALYSIS OF DISAGREEMENTS
# ---------------------------
# Flag semantic and pragmatic disagreements (assumes non-empty values indicate disagreement)
data['semantic_disagreement_flag'] = data['semantic disagreement'].notna() & (data['semantic disagreement'] != "")
data['pragmatic_disagreement_flag'] = data['pragmatic disagreement'].notna() & (data['pragmatic disagreement'] != "")

# Aggregate disagreement counts per study/session
disagreement_counts = data.groupby('studyId').agg(
    semantic_disagreements=('semantic_disagreement_flag', 'sum'),
    pragmatic_disagreements=('pragmatic_disagreement_flag', 'sum')
).reset_index()
print("\nDisagreement counts per study/session:")
print(disagreement_counts)

plt.figure(figsize=(8, 6))
bar_width = 0.35
x = range(len(disagreement_counts))
plt.bar(x, disagreement_counts['semantic_disagreements'], width=bar_width, color='skyblue', label='Semantic')
plt.bar([p + bar_width for p in x], disagreement_counts['pragmatic_disagreements'], width=bar_width, color='salmon',
        label='Pragmatic')
plt.xticks([p + bar_width / 2 for p in x], disagreement_counts['studyId'])
plt.xlabel('Study/Session')
plt.ylabel('Number of Disagreements')
plt.title('Disagreements per Session')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# 4. COMMAND-EXECUTION MATCHING ANALYSIS
# ---------------------------
# Create a new column indicating if the command was “satisfied” (assuming that when the 'to argument' equals the 'from argument', the command is satisfied)
data['command_satisfied'] = data['to argument [user or GPT]'] == data['from argument [user or GPT]']

# Assuming rows labeled as 'command' in MIDAS represent commands, aggregate satisfaction metrics by study/session.
command_satisfaction = data.groupby('studyId').agg(
    total_commands=('MIDAS', lambda x: (x == 'command').sum()),
    satisfied_commands=('command_satisfied', 'sum')
).reset_index()

# Calculate satisfaction rate (avoiding division by zero)
command_satisfaction['satisfaction_rate'] = command_satisfaction.apply(
    lambda row: row['satisfied_commands'] / row['total_commands'] if row['total_commands'] > 0 else 0,
    axis=1
)
print("\nCommand satisfaction metrics per study/session:")
print(command_satisfaction)

plt.figure(figsize=(8, 6))
sns.barplot(data=command_satisfaction, x='studyId', y='satisfaction_rate', palette='magma')
plt.title('Command Satisfaction Rate per Session')
plt.xlabel('Study/Session')
plt.ylabel('Satisfaction Rate')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ---------------------------
# 5. ANALYSIS OF 'POINT TO' AND 'POINTS BACK'
# ---------------------------
# Check how often the "point to" and "points back" columns match.
data['pointing_match'] = data['point to'] == data['points back']

pointing_matches = data.groupby('studyId')['pointing_match'].sum().reset_index(name='match_count')
print("\n'Point to' and 'Points back' match counts per study/session:")
print(pointing_matches)

plt.figure(figsize=(8, 6))
sns.barplot(data=pointing_matches, x='studyId', y='match_count', palette='cool')
plt.title('"Point To" and "Points Back" Matches per Session')
plt.xlabel('Study/Session')
plt.ylabel('Match Count')
plt.tight_layout()
plt.show()

# ---------------------------
# 6. FURTHER ANALYSES (Additional Analysis)
# ---------------------------
# 6.1 Turn-Level Dynamics: Visualize the sequence of MIDAS acts per chat.
# If no explicit turn order exists, we create one based on the order in the DataFrame.
data['turn'] = data.groupby('chat_id').cumcount() + 1

# Aggregate counts of MIDAS acts at each turn (across all chats)
turn_midas = data.groupby('turn')['MIDAS'].value_counts().unstack().fillna(0)
print("\nTurn-level distribution of MIDAS acts:")
print(turn_midas)

plt.figure(figsize=(10, 6))
turn_midas.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Stacked Bar Chart of MIDAS Acts by Turn Number')
plt.xlabel('Turn Number')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 6.2 Sequential Pattern Analysis: Transition probabilities between MIDAS acts.
# First, sort data by chat_id and turn.
data_sorted = data.sort_values(by=['chat_id', 'turn'])


# Define a helper function to compute transitions in a list
def compute_transitions(midas_sequence):
    return list(zip(midas_sequence, midas_sequence[1:]))


transition_list = []
for chat, group in data_sorted.groupby('chat_id'):
    midas_seq = group['MIDAS'].tolist()
    transitions = compute_transitions(midas_seq)
    transition_list.extend(transitions)

transition_df = pd.DataFrame(transition_list, columns=['from', 'to'])
transition_matrix = pd.crosstab(transition_df['from'], transition_df['to'], normalize='index')
print("\nTransition Probability Matrix for MIDAS Acts:")
print(transition_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(transition_matrix, annot=True, cmap='Blues')
plt.title('Transition Probabilities between MIDAS Acts')
plt.tight_layout()
plt.show()

# 6.3 Clustering Participants Based on Interaction Patterns
# Create participant-level features: total messages, total commands, satisfied commands,
# semantic disagreements, and pragmatic disagreements.
participant_features = data.groupby('pid').agg(
    total_messages=('text', 'count'),
    total_commands=('MIDAS', lambda x: (x == 'command').sum()),
    satisfied_commands=('command_satisfied', 'sum'),
    semantic_disagreements=('semantic_disagreement_flag', 'sum'),
    pragmatic_disagreements=('pragmatic_disagreement_flag', 'sum')
).reset_index()

# Calculate satisfaction rate (and disagreement rates) per participant.
participant_features['satisfaction_rate'] = participant_features.apply(
    lambda row: row['satisfied_commands'] / row['total_commands'] if row['total_commands'] > 0 else 0,
    axis=1
)
participant_features['semantic_disagreement_rate'] = participant_features['semantic_disagreements'] / \
                                                     participant_features['total_messages']
participant_features['pragmatic_disagreement_rate'] = participant_features['pragmatic_disagreements'] / \
                                                      participant_features['total_messages']

print("\nParticipant Features for Clustering:")
print(participant_features)

# Select features for clustering.
features = ['satisfaction_rate', 'semantic_disagreement_rate', 'pragmatic_disagreement_rate', 'total_messages']
X = participant_features[features]

# Standardize the features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster participants using KMeans (here, using 2 clusters as an example)
kmeans = KMeans(n_clusters=2, random_state=42)
participant_features['cluster'] = kmeans.fit_predict(X_scaled)

print("\nParticipant Features with Cluster Assignment:")
print(participant_features[['pid', 'cluster'] + features])

plt.figure(figsize=(8, 6))
sns.scatterplot(data=participant_features, x='satisfaction_rate', y='semantic_disagreement_rate',
                hue='cluster', palette='Set1', s=100)
plt.title('Clustering Participants based on Satisfaction and Semantic Disagreement Rates')
plt.xlabel('Command Satisfaction Rate')
plt.ylabel('Semantic Disagreement Rate')
plt.tight_layout()
plt.show()

# 6.4 Latency Analysis (if a 'timestamp' column is available)
# This analysis computes the time difference (latency) between a command and the next turn.
if 'timestamp' in data.columns:
    # Convert the 'timestamp' column to datetime format if it is not already.
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    latencies = []
    # Ensure data is sorted by chat and turn.
    for chat, group in data_sorted.groupby('chat_id'):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            if group.loc[i, 'MIDAS'] == 'command':
                # Compute latency in seconds between the command and the following turn
                delta = (group.loc[i + 1, 'timestamp'] - group.loc[i, 'timestamp']).total_seconds()
                latencies.append(delta)

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage latency between command issuance and the next turn: {avg_latency:.2f} seconds")

        plt.figure(figsize=(8, 6))
        sns.histplot(latencies, bins=20, kde=True, color='purple')
        plt.title('Latency Distribution between Command Issuance and Next Turn')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo latency data computed: please ensure the 'timestamp' column exists and is properly formatted.")
else:
    print("\nNo 'timestamp' column available for latency analysis.")
