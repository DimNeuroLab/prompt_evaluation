import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and Inspect Data ---

# Read the CSV file containing the term-by-term coded dialogue
df = pd.read_csv("G:\\My Drive\\projects\\GPTInteractionAnnotationVonNeumidas\\ANNOTATIONS _FINAL_SOLMAZ.tsv",sep='\t')

# Print out the first few rows and column names to verify the structure
print("Columns in the CSV:", df.columns.tolist())
print(df.head())

# (Optional) Ensure that the session column is numeric
df['studyId'] = pd.to_numeric(df['studyId'], errors='coerce')


# --- 2. Compute Session-Level Metrics ---

# Example: For each session, calculate:
# - Average friendliness
# - Total number of commands
# - Number of commands with a 'satisfied' status
# - Command satisfaction rate (satisfied commands / total commands)
# need to check the success column.. other file
# def compute_session_metrics(group):
#     total_commands = (group['MIDAS'] == 'command').sum()
#     satisfied_commands = ((group['MIDAS'] == 'command') &
#                           (group['command_status'] == 'satisfied')).sum()
#
#     return pd.Series({
#         'avg_friendliness': group['friendliness'].mean(),
#         'num_commands': total_commands,
#         'num_satisfied': satisfied_commands,
#         'command_satisfaction_rate': satisfied_commands / total_commands if total_commands > 0 else np.nan
#     })
#
#
# session_metrics = df.groupby('studyId').apply(compute_session_metrics).reset_index()
# print("\nSession-Level Metrics:")
# print(session_metrics)

# --- 3. Plotting the Metrics ---

sns.set(style="whitegrid")

# # Plot 1: Average Friendliness over Sessions
#need to be frequency of social related speech acts
# plt.figure(figsize=(8, 6))
# sns.lineplot(x='session', y='avg_friendliness', data=session_metrics, marker='o')
# plt.title('Average Friendliness Over Sessions')
# plt.xlabel('Session')
# plt.ylabel('Average Friendliness')
# plt.xticks(session_metrics['session'])
# plt.tight_layout()
# plt.show()

# Plot 2: Command Satisfaction Rate over Sessions
#comes from the success evaluation
# plt.figure(figsize=(8, 6))
# sns.lineplot(x='session', y='command_satisfaction_rate', data=session_metrics, marker='o')
# plt.title('Command Satisfaction Rate Over Sessions')
# plt.xlabel('Session')
# plt.ylabel('Satisfaction Rate')
# plt.xticks(session_metrics['session'])
# plt.tight_layout()
# plt.show()

# --- 4. Command Frequency by Participant ---

# Filter for command acts only
commands_df = df[df['MIDAS'] == 'command']

# Count commands per session and participant
command_by_participant = commands_df.groupby(['studyId', 'pid']).size().reset_index(name='command_count')
print("\nCommand Count by Participant:")
print(command_by_participant)

plt.figure(figsize=(8, 6))
sns.barplot(x='studyId', y='command_count', hue='pid', data=command_by_participant)
plt.title('Command Count by Participant per Session')
plt.xlabel('Session')
plt.ylabel('Command Count')
plt.tight_layout()
plt.show()

# --- 5. Error Analysis (if error_type is provided) ---

# Only include rows with an error type
if 'error_type' in df.columns:
    error_counts = df[df['error_type'].notnull()].groupby(['session', 'error_type']).size().reset_index(
        name='error_count')
    print("\nError Counts by Session and Error Type:")
    print(error_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='session', y='error_count', hue='error_type', data=error_counts)
    plt.title('Error Types Distribution by Session')
    plt.xlabel('Session')
    plt.ylabel('Error Count')
    plt.tight_layout()
    plt.show()

# --- 6. Turn-Level Analysis within a Single Session ---

# For a closer look at one session (e.g., Session 1), plot friendliness over turns per participant
session1 = df[df['studyId'] == 1]

if 'turn' in session1.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='turn', y='friendliness', hue='participant', data=session1, marker='o')
    plt.title('Friendliness Over Turns in Session 1')
    plt.xlabel('Turn')
    plt.ylabel('Friendliness')
    plt.tight_layout()
    plt.show()

# --- 7. Additional Analyses ---

# You might also consider:
# - Computing latency metrics if you have a 'timestamp' column (e.g., time between a command and its execution)
# - Analyzing transitions between dialogue acts with sequence models
# - Clustering sessions or participants based on multiple metrics

# Example: If timestamps exist and you want to compute time between turns
if 'timestamp' in df.columns:
    # Convert timestamp column to datetime (adjust format as needed)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['session', 'turn'])  # ensure proper order

    # Compute time difference (in seconds) between consecutive turns for each session
    df['time_diff'] = df.groupby('session')['timestamp'].diff().dt.total_seconds()

    # Plot average time difference per session
    time_diff_metrics = df.groupby('session')['time_diff'].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.lineplot(x='session', y='time_diff', data=time_diff_metrics, marker='o')
    plt.title('Average Time Difference Between Turns per Session')
    plt.xlabel('Session')
    plt.ylabel('Average Time Difference (s)')
    plt.tight_layout()
    plt.show()
