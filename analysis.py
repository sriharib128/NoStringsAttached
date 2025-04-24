# %%
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# File paths
debiased_scores_path = "final_bias_scores_debiasing.json"
biased_scores_path = "final_bias_scores_no_debiasing.json"

# Function to read JSON file
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Read the JSON files
debiased_scores = read_json(debiased_scores_path)
biased_scores = read_json(biased_scores_path)

# Calculate the average bias score for each profession
def calculate_average_bias_scores(scores):
    average_bias_scores = {}
    for profession, score in scores.items():
        average_bias_scores[profession] = sum(score) / len(score) if isinstance(score, list) else score
    return average_bias_scores

average_debiased_scores = calculate_average_bias_scores(debiased_scores)
average_biased_scores = calculate_average_bias_scores(biased_scores)

# Calculate overall bias scores for all professions
def calculate_overall_bias_score(average_scores):
    all_scores = list(average_scores.values())
    return sum(all_scores) / len(all_scores)

overall_debiased_score = calculate_overall_bias_score(average_debiased_scores)
overall_biased_score = calculate_overall_bias_score(average_biased_scores)

# Print overall bias scores
print(f"Overall Average Debiased Score: {overall_debiased_score}")
print(f"Overall Average Biased Score: {overall_biased_score}")

# %%

# Create a DataFrame to hold the scores for plotting
professions = list(average_debiased_scores.keys())
debiased_values = list(average_debiased_scores.values())
biased_values = [average_biased_scores.get(profession, 'N/A') for profession in professions]

# Create a DataFrame
df = pd.DataFrame({
    'Profession': professions,
    'Average Debiased Score': debiased_values,
    'Average Biased Score': biased_values
})

# Set the profession as the index for easy plotting
df.set_index('Profession', inplace=True)

# Plot the comparison for each profession
plt.figure(figsize=(15, 10))
df.plot(kind='bar', width=0.8)
plt.title('Average Debiased vs Biased Scores for Professions')
plt.ylabel('Average Bias Score')
plt.xlabel('Profession')
plt.xticks(rotation=90)
plt.tight_layout()  # Adjust layout to prevent overlap of labels
plt.legend(title='Score Type')
plt.show()


# %%
# Initialize an empty dictionary to store the ID to name mapping
id_name_mapping = {}

# Open the file and read it line by line
with open('FB15k_mid2name.txt', 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace
        line = line.strip()
        
        # Split the line into ID and name
        id_value, name = line.split('\t')
        
        # Add the ID and name to the dictionary
        id_name_mapping[id_value] = name

# Print the dictionary to see the result
id_name_mapping


# Assuming professions, debiased_values, and biased_values are already defined
# Create a DataFrame
new_df = pd.DataFrame({
    'Profession': professions,
    'Before Debiasing': biased_values,
    "After Debiasing" : debiased_values,
    'Average Debiased Score': (np.array(biased_values) - np.array(debiased_values))
})

# Add the Name column using the id_name_mapping dictionary
new_df['Name'] = new_df['Profession'].map(id_name_mapping)

# Sort the DataFrame in descending order of debiased score
new_df = new_df.sort_values(by='Average Debiased Score', ascending=False)
new_df.head(10)

# %%
import numpy as np
import pandas as pd

# Assuming professions, debiased_values, and biased_values are already defined
# Create a DataFrame
new_df = pd.DataFrame({
    'Profession': professions,
    'Average Debiased Score': (np.array(debiased_values) - np.array(biased_values))
})


# Sort the DataFrame in descending order of debiased score
new_df = new_df.sort_values(by='Average Debiased Score', ascending=False)

# Display the sorted DataFrame
print(new_df)


