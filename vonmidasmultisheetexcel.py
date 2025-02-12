import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. LOAD THE DATA FROM THE XLSX
# ---------------------------

# Path to your Excel file
#xlsx_file = "interactions.xlsx"
xlsx_file = "G:\\My Drive\\projects\\GPTInteractionAnnotationVonNeumidas\\ANNOTATIONS _FINAL_SOLMAZ.xlsx"#'von_neumidas_interactions.xlsx'

# Load the Excel file and get the list of sheets (each sheet represents one interaction)
xls = pd.ExcelFile(xlsx_file)
sheet_names = xls.sheet_names

# Read each sheet into a DataFrame and add a column with the sheet name (optional)
dfs = []
for sheet in sheet_names:
    df_sheet = pd.read_excel(xlsx_file, sheet_name=sheet)
    df_sheet['sheet'] = sheet  # Optional: keep track of which sheet the row came from
    dfs.append(df_sheet)

# Concatenate all sheets into one DataFrame for analysis
data = pd.concat(dfs, ignore_index=True)

# Convert key columns to strings (if not already)
data['pid'] = data['pid'].astype(str)
data['studyId'] = data['studyId'].astype(str)

# Inspect the data
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

# For semantic and pragmatic disagreements, we assume that a non-empty value (or a flag value) indicates a disagreement.
# Here, we create Boolean flags (True if there is a disagreement).
data['semantic_disagreement_flag'] = data['semantic disagreement'].notna() & (data['semantic disagreement'] != "")
data['pragmatic_disagreement_flag'] = data['pragmatic disagreement'].notna() & (data['pragmatic disagreement'] != "")

# Aggregate the number of disagreements per study/session
disagreement_counts = data.groupby('studyId').agg(
    semantic_disagreements=('semantic_disagreement_flag', 'sum'),
    pragmatic_disagreements=('pragmatic_disagreement_flag', 'sum')
).reset_index()
print("\nDisagreement counts per study/session:")
print(disagreement_counts)

# Plotting disagreements side by side
plt.figure(figsize=(8, 6))
bar_width = 0.35
x = range(len(disagreement_counts))

plt.bar(x, disagreement_counts['semantic_disagreements'], width=bar_width, color='skyblue', label='Semantic')
plt.bar([p + bar_width for p in x], disagreement_counts['pragmatic_disagreements'], width=bar_width, color='salmon', label='Pragmatic')
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

# One way to analyze command matching is to look at the relationship between the 'to argument [user or GPT]' and
# the 'from argument [user or GPT]'. For instance, you might expect that for a well-satisfied command these two values match.
# (Adjust this logic based on your coding scheme.)

# Create a new column that indicates whether the command seems to have been “satisfied”
data['command_satisfied'] = data['to argument [user or GPT]'] == data['from argument [user or GPT]']

# Now, assuming that a row with a MIDAS label of "command" indicates that a command was issued,
# we can aggregate satisfaction metrics by study/session.
# (If your MIDAS column uses a different label to indicate commands, adjust the condition below.)
command_satisfaction = data.groupby('studyId').agg(
    total_commands=('MIDAS', lambda x: (x == 'command').sum()),
    satisfied_commands=('command_satisfied', 'sum')
).reset_index()

# Calculate satisfaction rate (handle division by zero if no commands are present)
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

# If your schema uses the "point to" and "points back" columns to capture reference matching,
# you might want to check how often these match.
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
# 6. FURTHER ANALYSES (OPTIONAL)
# ---------------------------
# For example, you could:
# - Analyze turn-level dynamics (e.g., plotting the sequence of MIDAS acts per chat).
# - Perform sequential pattern mining on the dialogue acts.
# - Cluster participants based on adaptation patterns across sessions.
# - Analyze latency between command issuance and correction or execution (if timestamps are available).
#
# These additional analyses will depend on the specifics of your data and research questions.
