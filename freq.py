# %%
from collections import defaultdict

# Read MID to name mapping from FB15k_mid2name.txt
mid_to_name = {}
with open('FB15k_mid2name.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            mid = parts[0]
            name = parts[1]
            mid_to_name[mid] = name

# Initialize counters for male and female professions
male_jobs = defaultdict(int)
female_jobs = defaultdict(int)

# Process gen2prof_fair_all.txt to count professions by gender
with open('gen2prof_fair_all.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        gender_mid = parts[0]
        profession_mid = parts[4]
        
        gender_name = mid_to_name.get(gender_mid, None)
        profession_name = mid_to_name.get(profession_mid, None)
        
        if gender_name == 'Male' and profession_name:
            male_jobs[profession_name] += 1
        elif gender_name == 'Female' and profession_name:
            female_jobs[profession_name] += 1

# Collect all unique professions
all_professions = set(male_jobs.keys()).union(set(female_jobs.keys()))

# Calculate male/female ratios for each profession
profession_ratios = []
for prof in all_professions:
    male = male_jobs.get(prof, 0)
    female = female_jobs.get(prof, 0)
    if female == 0:
        ratio = float('inf')
    else:
        ratio = male / female
    profession_ratios.append((prof, male, female, ratio))

# Sort professions by ratio descending for male-dominated and ascending for female-dominated
profession_ratios_sorted_desc = sorted(profession_ratios, key=lambda x: (-x[3], x[0]))
profession_ratios_sorted_asc = sorted(profession_ratios, key=lambda x: (x[3], x[0]))

# Get top 10 entries for both categories
top_male_dominated = profession_ratios_sorted_desc[:10]
top_female_dominated = profession_ratios_sorted_asc[:10]

# Print the results
print("Top 10 Male-Dominated Professions (Highest Male/Female Ratio):")
print("Profession\tMale\tFemale\tRatio")
for entry in top_male_dominated:
    prof, male, female, ratio = entry
    ratio_str = "∞" if ratio == float('inf') else f"{ratio:.2f}"
    print(f"{prof}\t{male}\t{female}\t{ratio_str}")

print("\nTop 10 Female-Dominated Professions (Lowest Male/Female Ratio):")
print("Profession\tMale\tFemale\tRatio")
for entry in top_female_dominated:
    prof, male, female, ratio = entry
    ratio_str = f"{ratio:.2f}"
    print(f"{prof}\t{male}\t{female}\t{ratio_str}")

# %%
from collections import defaultdict

# Read MID to name mapping from FB15k_mid2name.txt
mid_to_name = {}
with open('FB15k_mid2name.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            mid = parts[0]
            name = parts[1]
            mid_to_name[mid] = name

# Initialize counters for male and female professions
male_jobs = defaultdict(int)
female_jobs = defaultdict(int)

# Process gen2prof_fair_all.txt to count professions by gender
with open('gen2prof_fair_all.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        gender_mid = parts[0]
        profession_mid = parts[4]
        
        gender_name = mid_to_name.get(gender_mid, None)
        profession_name = mid_to_name.get(profession_mid, None)
        
        if gender_name == 'Male' and profession_name:
            male_jobs[profession_name] += 1
        elif gender_name == 'Female' and profession_name:
            female_jobs[profession_name] += 1

# Collect all unique professions and filter by total count
profession_ratios = []
for prof in set(male_jobs.keys()).union(set(female_jobs.keys())):
    male = male_jobs.get(prof, 0)
    female = female_jobs.get(prof, 0)
    total = male + female
    
    # # Skip professions with total count < 20
    if total < 1:
        continue
    
    if female == 0:
        ratio = float('inf')
    else:
        ratio = male / female
    
    if male==0:
        f_ratio = float('inf')
    else:
        f_ratio = female/male
    profession_ratios.append((prof, male, female, ratio,f_ratio))

# Sort professions by ratio descending for male-dominated and ascending for female-dominated
profession_ratios_sorted_desc = sorted(profession_ratios, key=lambda x: (-x[3], x[0]))
profession_ratios_sorted_asc = sorted(profession_ratios, key=lambda x: (-x[4], x[0]))

# Get top 10 entries for both categories
top_male_dominated = profession_ratios_sorted_desc[:10]
top_female_dominated = profession_ratios_sorted_asc[:10]

# Print the results
print("Top 10 Male-Dominated Professions (Highest Male/Female Ratio, Total ≥20):")
print("Profession\tMale\tFemale\tRatio")
for entry in top_male_dominated:
    prof, male, female, ratio,f_ratio = entry
    ratio_str = "∞" if ratio == float('inf') else f"{ratio:.2f}"
    print(f"{prof}\t{male}\t{female}\t{ratio_str}")

print("\nTop 10 Female-Dominated Professions (Lowest Male/Female Ratio, Total ≥20):")
print("Profession\tMale\tFemale\tRatio")
for entry in top_female_dominated:
    prof, male, female, m_ratio,ratio = entry
    ratio_str = f"{ratio:.2f}"
    print(f"{prof}\t{male}\t{female}\t{ratio_str}")

# %%
from collections import defaultdict

# Read MID to name mapping from gen2prof_fair_all.txt
mid_to_name = {}
with open('FB15k_mid2name.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            mid = parts[0]
            name = parts[1]
            mid_to_name[mid] = name


# Initialize counters for male and female professions
male_jobs = defaultdict(int)
female_jobs = defaultdict(int)


# Process FB15k_mid2name.txt to count professions by gender
with open('gen2prof_fair_all.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        gender_mid = parts[0]
        profession_mid = parts[4]
        
        gender_name = mid_to_name.get(gender_mid, None)
        profession_name = mid_to_name.get(profession_mid, None)
        # print(gender_name,profession_name)
        if gender_name == 'Male' and profession_name:
            male_jobs[profession_name] += 1
        elif gender_name == 'Female' and profession_name:
            female_jobs[profession_name] += 1
    

# Function to sort and return top N entries
def get_top_jobs(job_dict, top_n=20):
    sorted_jobs = sorted(job_dict.items(), key=lambda x: (-x[1], x[0]))
    return sorted_jobs[:top_n]

# Get top 10 for each (adjust top_n as needed)
top_male = get_top_jobs(male_jobs)
top_female = get_top_jobs(female_jobs)

# Print tables
print("Top Male Jobs:")
print("Profession\tCount")
for job, count in top_male:
    print(f"{job}\t{count}")

print("\nTop Female Jobs:")
print("Profession\tCount")
for job, count in top_female:
    print(f"{job}\t{count}")

# %%



