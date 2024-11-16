import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('dataset.csv')
df = df.dropna()
df['label'] = df['EF'].apply(lambda x: 1 if str(x).isdigit() and int(x) <= 50 else 0)
df.drop('EF', axis = 1)
df.drop('HTN', axis = 1)
df['GENDER'] = df['GENDER'].apply(lambda x: 1 if x == 'Female' else 0)
df.head()

X = df[['AGE', 'GENDER', 'DM', 'AF']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

input_data = np.array([[60, 0, 1, 1]])  
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
probability = model.predict_proba(input_data_scaled)

print(f"Prediction: {prediction[0]} (EF {'< 50%' if prediction[0] == 1 else '≥ 50%'})")
print(f"Probability: {probability}")
