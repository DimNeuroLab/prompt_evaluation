import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 1. Define a fixed color mapping for MIDAS speech acts.
# Adjust keys and colors as needed.
midas_colors = {
    'command': '#1f77b4',  # Blue
    'question': '#ff7f0e',  # Orange
    'assertion': '#2ca02c',  # Green
    'clarification': '#d62728',  # Red
    'greeting': '#9467bd',  # Purple
    'farewell': '#8c564b',  # Brown,
    # Add more keys if desired; missing ones will get default colors.
}

# ---------------------------
# 2. LOAD THE DATA FROM THE XLSX
# ---------------------------
xlsx_file = "G:\\My Drive\\projects\\GPTInteractionAnnotationVonNeumidas\\ANNOTATIONS _FINAL_SOLMAZ.xlsx"#'von_neumidas_interactions.xlsx'

# Load the Excel file and get the list of sheets (each sheet represents one interaction)
xls = pd.ExcelFile(xlsx_file)
sheet_names = xls.sheet_names

dfs = []
for sheet in sheet_names:
    df_sheet = pd.read_excel(xlsx_file, sheet_name=sheet)
    df_sheet['sheet'] = sheet  # Optional: track which sheet each row came from
    dfs.append(df_sheet)

# Concatenate all sheets into one DataFrame for analysis
data = pd.concat(dfs, ignore_index=True)

# Ensure key columns are strings
data['pid'] = data['pid'].astype(str)
data['studyId'] = data['studyId'].astype(str)

print("Data head:")
print(data.head())
print("\nData info:")
print(data.info())

# ---------------------------
# 3. Create a complete palette for MIDAS acts
# ---------------------------
# Get the unique MIDAS acts in the data.
unique_midas = data['MIDAS'].unique()

# Start with the provided colors in midas_colors; assign default colors for missing keys.
complete_midas_colors = {}
missing_keys = []
for key in unique_midas:
    if key in midas_colors:
        complete_midas_colors[key] = midas_colors[key]
    else:
        missing_keys.append(key)

# For any missing keys, generate default colors using Seaborn’s husl palette.
if missing_keys:
    default_palette = sns.color_palette("husl", len(missing_keys)).as_hex()
    for i, key in enumerate(missing_keys):
        complete_midas_colors[key] = default_palette[i]

# (Optional) Print the complete palette to check:
print("\nComplete MIDAS palette:")
print(complete_midas_colors)

# ---------------------------
# 4. Create role-based subsets
# ---------------------------
data_user = data[data['role'].str.lower() == 'user']
data_assistant = data[data['role'].str.lower().isin(['assistant', 'system'])]

# ---------------------------
# 5. BASIC DESCRIPTIVE ANALYSES BY ROLE
# ---------------------------
# 5.1 Count the number of messages per study/session by role
session_message_counts_role = data.groupby(['studyId', 'role'])['text'].count().reset_index(name='message_count')
print("\nMessage counts per study/session by role:")
print(session_message_counts_role)

plt.figure(figsize=(10, 6))
sns.barplot(data=session_message_counts_role, x='studyId', y='message_count', hue='role', palette='viridis')
plt.title('Number of Messages per Study/Session by Role')
plt.xlabel('Study/Session')
plt.ylabel('Message Count')
plt.tight_layout()
plt.show()

# 5.2 Distribution of MIDAS speech acts across sessions by role
midas_distribution_role = data.groupby(['studyId', 'role', 'MIDAS'])['text'].count().reset_index(name='count')
print("\nMIDAS distribution per study/session by role:")
print(midas_distribution_role)

plt.figure(figsize=(14, 8))
sns.barplot(data=midas_distribution_role, x='studyId', y='count', hue='MIDAS',
            palette=complete_midas_colors, ci=None)
plt.title('Distribution of MIDAS Speech Acts per Session (All Roles)')
plt.xlabel('Study/Session')
plt.ylabel('Count')
plt.legend(title='MIDAS Act', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Optionally, separate plots per role:
for role, subset in data.groupby('role'):
    plt.figure(figsize=(10, 6))
    role_dist = subset.groupby(['studyId', 'MIDAS'])['text'].count().reset_index(name='count')
    sns.barplot(data=role_dist, x='studyId', y='count', hue='MIDAS', palette=complete_midas_colors, ci=None)
    plt.title(f'Distribution of MIDAS Speech Acts per Session for Role: {role}')
    plt.xlabel('Study/Session')
    plt.ylabel('Count')
    plt.legend(title='MIDAS Act', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ---------------------------
# 6. ANALYSIS OF DISAGREEMENTS BY ROLE
# ---------------------------
# Flag semantic and pragmatic disagreements (assumes non-empty values indicate disagreement)
data['semantic_disagreement_flag'] = data['semantic disagreement'].notna() & (data['semantic disagreement'] != "")
data['pragmatic_disagreement_flag'] = data['pragmatic disagreement'].notna() & (data['pragmatic disagreement'] != "")

disagreement_counts_role = data.groupby(['studyId', 'role']).agg(
    semantic_disagreements=('semantic_disagreement_flag', 'sum'),
    pragmatic_disagreements=('pragmatic_disagreement_flag', 'sum')
).reset_index()
print("\nDisagreement counts per study/session by role:")
print(disagreement_counts_role)

plt.figure(figsize=(12, 6))
sns.barplot(data=disagreement_counts_role, x='studyId', y='semantic_disagreements', hue='role', palette='Blues',
            ci=None)
plt.title('Semantic Disagreements per Session by Role')
plt.xlabel('Study/Session')
plt.ylabel('Semantic Disagreement Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=disagreement_counts_role, x='studyId', y='pragmatic_disagreements', hue='role', palette='Reds',
            ci=None)
plt.title('Pragmatic Disagreements per Session by Role')
plt.xlabel('Study/Session')
plt.ylabel('Pragmatic Disagreement Count')
plt.tight_layout()
plt.show()

# ---------------------------
# 7. COMMAND-EXECUTION MATCHING ANALYSIS BY ROLE
# ---------------------------
# Create a column indicating if the command was “satisfied”
data['command_satisfied'] = data['to argument [user or GPT]'] == data['from argument [user or GPT]']

# Compute satisfaction metrics for rows labeled as 'command' in MIDAS.
for role_name, role_data in data.groupby('role'):
    cmd_data = role_data[role_data['MIDAS'] == 'command']
    total_commands = cmd_data.shape[0]
    satisfied_commands = cmd_data['command_satisfied'].sum()
    satisfaction_rate = satisfied_commands / total_commands if total_commands > 0 else 0
    print(
        f"\nRole: {role_name} -- Total Commands: {total_commands}, Satisfied: {satisfied_commands}, Satisfaction Rate: {satisfaction_rate:.2f}")

command_satisfaction_role = data[data['MIDAS'] == 'command'].groupby(['studyId', 'role']).agg(
    total_commands=('MIDAS', 'count'),
    satisfied_commands=('command_satisfied', 'sum')
).reset_index()

command_satisfaction_role['satisfaction_rate'] = command_satisfaction_role.apply(
    lambda row: row['satisfied_commands'] / row['total_commands'] if row['total_commands'] > 0 else 0,
    axis=1
)
print("\nCommand satisfaction metrics per study/session by role:")
print(command_satisfaction_role)

plt.figure(figsize=(10, 6))
sns.barplot(data=command_satisfaction_role, x='studyId', y='satisfaction_rate', hue='role', palette='magma', ci=None)
plt.title('Command Satisfaction Rate per Session by Role')
plt.xlabel('Study/Session')
plt.ylabel('Satisfaction Rate')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ---------------------------
# 8. ANALYSIS OF 'POINT TO' AND 'POINTS BACK' BY ROLE
# ---------------------------
data['pointing_match'] = data['point to'] == data['points back']
pointing_matches_role = data.groupby(['studyId', 'role'])['pointing_match'].sum().reset_index(name='match_count')
print("\n'Point to' and 'Points back' match counts per study/session by role:")
print(pointing_matches_role)

plt.figure(figsize=(10, 6))
sns.barplot(data=pointing_matches_role, x='studyId', y='match_count', hue='role', palette='cool', ci=None)
plt.title('"Point To" and "Points Back" Matches per Session by Role')
plt.xlabel('Study/Session')
plt.ylabel('Match Count')
plt.tight_layout()
plt.show()

# ---------------------------
# 9. FURTHER ANALYSES (Additional Analysis)
# ---------------------------
# 9.1 Turn-Level Dynamics: Visualize the sequence of MIDAS acts per chat.
data['turn'] = data.groupby('chat_id').cumcount() + 1

# For overall turn-level dynamics, group by turn and role.
turn_midas = data.groupby(['turn', 'role'])['MIDAS'].value_counts().unstack().fillna(0).reset_index()
print("\nTurn-level distribution of MIDAS acts by role:")
print(turn_midas.head())

# Plot stacked bar charts for each role using our fixed color mapping.
for role_name, role_data in data.groupby('role'):
    turn_role = role_data.groupby('turn')['MIDAS'].value_counts().unstack().fillna(0)
    plt.figure(figsize=(10, 6))
    # Determine the order of MIDAS acts
    midas_order = list(turn_role.columns)
    # Map colors using our complete palette; if any key is missing, default to gray.
    colors = [complete_midas_colors.get(label, '#333333') for label in midas_order]
    turn_role.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6))
    plt.title(f'Stacked Bar Chart of MIDAS Acts by Turn (Role: {role_name})')
    plt.xlabel('Turn Number')
    plt.ylabel('Count')
    plt.legend(title='MIDAS Act', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# 9.2 Sequential Pattern Analysis: Transition probabilities between MIDAS acts.
def compute_transitions(midas_sequence):
    return list(zip(midas_sequence, midas_sequence[1:]))


data_sorted = data.sort_values(by=['chat_id', 'turn'])
transition_list = []
for chat, group in data_sorted.groupby('chat_id'):
    midas_seq = group['MIDAS'].tolist()
    transitions = compute_transitions(midas_seq)
    transition_list.extend(transitions)

transition_df = pd.DataFrame(transition_list, columns=['from', 'to'])
transition_matrix = pd.crosstab(transition_df['from'], transition_df['to'], normalize='index')
print("\nOverall Transition Probability Matrix for MIDAS Acts:")
print(transition_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(transition_matrix, annot=True, cmap='Blues')
plt.title('Overall Transition Probabilities between MIDAS Acts')
plt.tight_layout()
plt.show()

# Transition matrix for each role separately:
for role_name, role_data in data.groupby('role'):
    role_sorted = role_data.sort_values(by=['chat_id', 'turn'])
    transition_list_role = []
    for chat, group in role_sorted.groupby('chat_id'):
        midas_seq = group['MIDAS'].tolist()
        if len(midas_seq) > 1:
            transitions = compute_transitions(midas_seq)
            transition_list_role.extend(transitions)
    if transition_list_role:
        transition_df_role = pd.DataFrame(transition_list_role, columns=['from', 'to'])
        transition_matrix_role = pd.crosstab(transition_df_role['from'], transition_df_role['to'], normalize='index')
        print(f"\nTransition Probability Matrix for MIDAS Acts for role: {role_name}")
        print(transition_matrix_role)
        plt.figure(figsize=(8, 6))
        sns.heatmap(transition_matrix_role, annot=True, cmap='Greens')
        plt.title(f'Transition Probabilities between MIDAS Acts (Role: {role_name})')
        plt.tight_layout()
        plt.show()

# 9.3 Clustering Participants Based on Interaction Patterns (Users only)
# Instead of using data_user (which might be outdated), we filter the full data
# at the time of grouping to ensure all computed columns are present.
participant_features = data[data['role'].str.lower() == 'user'].groupby('pid').agg(
    total_messages=('text', 'count'),
    total_commands=('MIDAS', lambda x: (x == 'command').sum()),
    satisfied_commands=('command_satisfied', 'sum'),
    semantic_disagreements=('semantic_disagreement_flag', 'sum'),
    pragmatic_disagreements=('pragmatic_disagreement_flag', 'sum')
).reset_index()

participant_features['satisfaction_rate'] = participant_features.apply(
    lambda row: row['satisfied_commands'] / row['total_commands'] if row['total_commands'] > 0 else 0,
    axis=1
)
participant_features['semantic_disagreement_rate'] = participant_features['semantic_disagreements'] / participant_features['total_messages']
participant_features['pragmatic_disagreement_rate'] = participant_features['pragmatic_disagreements'] / participant_features['total_messages']

print("\nParticipant Features for Clustering (Users only):")
print(participant_features)

features = ['satisfaction_rate', 'semantic_disagreement_rate', 'pragmatic_disagreement_rate', 'total_messages']
X = participant_features[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
participant_features['cluster'] = kmeans.fit_predict(X_scaled)

print("\nParticipant Features with Cluster Assignment:")
print(participant_features[['pid', 'cluster'] + features])

plt.figure(figsize=(8, 6))
sns.scatterplot(data=participant_features, x='satisfaction_rate', y='semantic_disagreement_rate',
                hue='cluster', palette='Set1', s=100)
plt.title('Clustering Participants (Users) based on Satisfaction and Semantic Disagreement Rates')
plt.xlabel('Command Satisfaction Rate')
plt.ylabel('Semantic Disagreement Rate')
plt.tight_layout()
plt.show()

# 9.4 Latency Analysis (if a 'timestamp' column is available)
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    latencies = []
    data_sorted = data.sort_values(by=['chat_id', 'turn'])

    # Assume a user command is followed by the assistant's response.
    for chat, group in data_sorted.groupby('chat_id'):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            if group.loc[i, 'role'].lower() == 'user' and group.loc[i, 'MIDAS'] == 'command':
                for j in range(i + 1, len(group)):
                    if group.loc[j, 'role'].lower() in ['assistant', 'system']:
                        delta = (group.loc[j, 'timestamp'] - group.loc[i, 'timestamp']).total_seconds()
                        latencies.append(delta)
                        break

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(
            f"\nAverage latency between user command issuance and the following assistant response: {avg_latency:.2f} seconds")

        plt.figure(figsize=(8, 6))
        sns.histplot(latencies, bins=20, kde=True, color='purple')
        plt.title('Latency Distribution between User Command and Assistant Response')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo latency data computed: please ensure the 'timestamp' column exists and is properly formatted.")
else:
    print("\nNo 'timestamp' column available for latency analysis.")
