import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

df = pd.read_csv("Breast Cancer METABRIC.csv")

# List of common identifiers (you may need to adjust this list)
df.drop(columns=['Patient ID'],inplace=True)

df_numeric = df.select_dtypes(include=[np.number])
df.fillna(df_numeric.mean(), inplace=True)

df = df.dropna(subset=['Overall Survival Status'])

categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    mode_value = df[column].mode()[0]
    df[column].fillna(mode_value,inplace=True)

# Function to remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

numeric_columns = df.select_dtypes(include=[np.number]).columns

# Apply outlier removal
for column in numeric_columns:
    df = remove_outliers(df, column)

pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define the features (X) and the output labels (y)
X = df.drop('Overall Survival Status', axis=1)
y = df['Overall Survival Status'] 

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align X_test with X_train columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Standardize the features (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

best_rf = RandomForestClassifier(random_state=42, **best_params)
best_rf.fit(X_train, y_train)

dump(best_rf, "best_rf.joblib")