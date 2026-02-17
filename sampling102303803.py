import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

url = "/content/Creditcard_data.csv"
try:
    data = pd.read_csv(url)
except Exception:
    data = pd.read_csv('Creditcard_data.csv')

X_full = data.drop('Class', axis=1)
y_full = data['Class']

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
)

smote_sampler = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote_sampler.fit_resample(X_train_raw, y_train_raw)

train_df = pd.concat([pd.DataFrame(X_train_bal, columns=X_full.columns), 
                      pd.DataFrame(y_train_bal, columns=['Class'])], axis=1)

population_size = len(train_df)
z_val = 1.96  
p_val = 0.5   
error_margin = 0.05

n_initial = (z_val**2 * p_val * (1 - p_val)) / (error_margin**2)
optimal_n = int(math.ceil(n_initial / (1 + (n_initial - 1) / population_size)))

s1 = train_df.sample(n=optimal_n, random_state=101) 

step_size = population_size // optimal_n
start_idx = np.random.randint(0, step_size)
s2 = train_df.iloc[start_idx::step_size].head(optimal_n) 

s3 = train_df.groupby('Class', group_keys=False).apply(
    lambda x: x.sample(n=int(optimal_n / 2), random_state=101)
) 

num_clusters = 10
train_df['Cluster_ID'] = np.random.randint(0, num_clusters, size=population_size)
chosen_clusters = np.random.choice(range(num_clusters), size=3, replace=False)
cluster_subset = train_df[train_df['Cluster_ID'].isin(chosen_clusters)].drop('Cluster_ID', axis=1)
s4 = cluster_subset.sample(n=optimal_n, random_state=101) 
train_df = train_df.drop('Cluster_ID', axis=1) 

s5 = train_df.sample(n=optimal_n, replace=True, random_state=101) 

sample_dict = {
    'Sampling1': s1,
    'Sampling2': s2,
    'Sampling3': s3,
    'Sampling4': s4,
    'Sampling5': s5
}

ml_models = {
    'M1': LogisticRegression(max_iter=500, random_state=42),
    'M2': DecisionTreeClassifier(random_state=42),
    'M3': RandomForestClassifier(random_state=42),
    'M4': SVC(random_state=42),
    'M5': KNeighborsClassifier()
}

results_grid = pd.DataFrame(index=ml_models.keys(), columns=sample_dict.keys())

for samp_name, samp_data in sample_dict.items():
    X_samp = samp_data.drop('Class', axis=1)
    y_samp = samp_data['Class']
    
    scaler = StandardScaler()
    X_samp_scaled = scaler.fit_transform(X_samp)
    X_test_scaled = scaler.transform(X_test_raw)
    
    for mod_name, model in ml_models.items():
        model.fit(X_samp_scaled, y_samp)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_raw, predictions)
        results_grid.loc[mod_name, samp_name] = round(accuracy * 100, 2)

print("\nAccuracies:")
print(results_grid)

print("\nBest samples:")
for mod in results_grid.index:
    best_samp = results_grid.loc[mod].astype(float).idxmax()
    best_acc = results_grid.loc[mod, best_samp]
    print(f"{mod}: {best_samp} ({best_acc}%)")
