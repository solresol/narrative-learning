import pandas as pd
import sqlite3
import uuid

# Load the dataset
titanic_df = pd.read_csv("/mnt/data/titanic.csv")

# 1. Convert PassengerId to a UUID
titanic_df['PatientID'] = [str(uuid.uuid4()) for _ in range(len(titanic_df))]

# 2. Convert Survived (0,1) -> (Failure, Success)
titanic_df['Outcome'] = titanic_df['Survived'].map({0: 'Failure', 1: 'Success'})

# 3. Convert Pclass (1,2,3) -> (Beta, Omicron, Delta)
titanic_df['Group'] = titanic_df['Pclass'].map({1: 'Beta', 2: 'Omicron', 3: 'Delta'})

# 4. Drop Name
titanic_df = titanic_df.drop(columns=['Name'])

# 5. Invert Sex
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 'female', 'female': 'male'})

# 6. Convert Age to Treatment Months (3 * Age), imputing missing values with mean
mean_age = titanic_df['Age'].mean()
titanic_df['Treatment Months'] = titanic_df['Age'].fillna(mean_age) * 3

titanic_df = titanic_df.drop(columns=['Age'])

# 7. Convert SibSp to "Genetic Class A Matches" (SibSp + 1)
titanic_df['Genetic Class A Matches'] = titanic_df['SibSp'] + 1

titanic_df = titanic_df.drop(columns=['SibSp'])

# 8. Convert Parch to "Genetic Class B Matches" (Parch + 1)
titanic_df['Genetic Class B Matches'] = titanic_df['Parch'] + 1

titanic_df = titanic_df.drop(columns=['Parch'])

# 9. Convert Fare to "TcQ mass" (1000 * Fare)
titanic_df['TcQ mass'] = titanic_df['Fare'] * 1000

titanic_df = titanic_df.drop(columns=['Fare'])

# 10. Drop Cabin
titanic_df = titanic_df.drop(columns=['Cabin'])

# 11. Change Embarked to "Cohort" (S,C,Q -> Melbourne, Delhi, Lisbon)
titanic_df['Cohort'] = titanic_df['Embarked'].map({'S': 'Melbourne', 'C': 'Delhi', 'Q': 'Lisbon'})

titanic_df = titanic_df.drop(columns=['Embarked'])

# Save the transformed dataset to a SQLite database
sqlite_path = "/mnt/data/titanic_medical.sqlite"
conn = sqlite3.connect(sqlite_path)
titanic_df.to_sql("medical_treatment_data", conn, if_exists="replace", index=False)
conn.close()
