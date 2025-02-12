import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# === Step 1: Load the Excel file ===
file_path = "G:\\My Drive\\projects\\GPTInteractionAnnotationVonNeumidas\\ANNOTATIONS _FINAL_SOLMAZ.xlsx"#'von_neumidas_interactions.xlsx'
# Reads all sheets; each key in the dictionary is a sheet name (i.e., one interaction)
all_sheets = pd.read_excel(file_path, sheet_name=None)

# === Step 2: Process Each Interaction (Sheet) ===
# We'll store summary statistics for each interaction in a list
interaction_summary = []

for sheet_name, df in all_sheets.items():
    print(f"Processing interaction: {sheet_name}")

    # --- Basic Cleaning & Setup ---
    # Fill missing values (customize as needed)
    df.fillna("", inplace=True)

    # Create a sequential turn counter (if not already present)
    df['turn'] = range(1, len(df) + 1)

    # --- Identify Commands ---
    # Here we assume that any row whose 'MIDAS' column contains "command" (ignoring case) is a command.
    df['is_command'] = df['MIDAS'].str.contains("command", case=False, na=False)

    # --- Convert Disagreement Columns to Numeric ---
    # We assume these columns are coded as 0/1 (or something convertible)
    df['semantic_disagreement_numeric'] = pd.to_numeric(df['semantic disagreement'], errors='coerce').fillna(0)
    df['pragmatic_disagreement_numeric'] = pd.to_numeric(df['pragmatic disagreement'], errors='coerce').fillna(0)

    # --- Determine Command Satisfaction ---
    # A command is "satisfied" if it is a command and has no semantic or pragmatic disagreements.
    df['command_satisfied'] = df.apply(
        lambda row: 1 if row['is_command'] and row['semantic_disagreement_numeric'] == 0 and row[
            'pragmatic_disagreement_numeric'] == 0 else 0,
        axis=1
    )

    # --- Compute Basic Metrics ---
    total_commands = df['is_command'].sum()
    satisfied_commands = df['command_satisfied'].sum()
    success_rate = satisfied_commands / total_commands if total_commands > 0 else np.nan
    total_semantic_disagreement = df['semantic_disagreement_numeric'].sum()
    total_pragmatic_disagreement = df['pragmatic_disagreement_numeric'].sum()
    role_counts = df['role'].value_counts().to_dict()  # e.g., how many rows per role (user vs. GPT)

    # Store the summary for this interaction
    interaction_summary.append({
        'sheet': sheet_name,
        'total_turns': len(df),
        'total_commands': total_commands,
        'satisfied_commands': satisfied_commands,
        'command_success_rate': success_rate,
        'total_semantic_disagreement': total_semantic_disagreement,
        'total_pragmatic_disagreement': total_pragmatic_disagreement,
        'role_counts': role_counts
    })

    # --- Plot: Command Satisfaction Timeline ---
    # This plot shows, for each turn, whether a command was satisfied (1) or not (0).
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x='turn', y='command_satisfied', hue='role', data=df, palette='deep')
    plt.title(f'Command Satisfaction Timeline for Interaction {sheet_name}')
    plt.xlabel("Turn")
    plt.ylabel("Command Satisfied (1=yes, 0=no)")
    plt.ylim(-0.1, 1.1)
    plt.show()

    # --- Additional: Print Frequency of Argument-related Labels ---
    print(f"'{sheet_name}' - 'to argument [user or GPT]' counts:")
    print(df['to argument [user or GPT]'].value_counts())
    print(f"'{sheet_name}' - 'from argument [user or GPT]' counts:")
    print(df['from argument [user or GPT]'].value_counts())
    print("-" * 50)

# === Step 3: Aggregate and Visualize Overall Statistics ===

# Create a DataFrame from the interaction summaries
summary_df = pd.DataFrame(interaction_summary)
print("Overall Interaction Summary:")
print(summary_df)

# Plot overall command success rates across interactions
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x='sheet', y='command_success_rate', palette='viridis')
plt.xlabel("Interaction (Sheet Name)")
plt.ylabel("Command Success Rate")
plt.title("Command Success Rate per Interaction")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Optional: Overall Correlation Between Disagreements ---
# Aggregate data from all sheets to see if semantic and pragmatic disagreements are related.
all_data = pd.concat(all_sheets.values(), ignore_index=True)
all_data['semantic_disagreement_numeric'] = pd.to_numeric(all_data['semantic disagreement'], errors='coerce').fillna(0)
all_data['pragmatic_disagreement_numeric'] = pd.to_numeric(all_data['pragmatic disagreement'], errors='coerce').fillna(
    0)
correlation = all_data[['semantic_disagreement_numeric', 'pragmatic_disagreement_numeric']].corr()
print("Correlation between Semantic and Pragmatic Disagreements:")
print(correlation)
