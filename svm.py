import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def runscript(csv_file_path):
    # Update this path to the location of your file
    df = pd.read_csv(csv_file_path)

    # Replace specific values with 'No'
    df = df.replace({'No phone service': 'No', 'No internet service': 'No'})

    # Convert string values to integers or floats, or factorize if not possible
    def str_to_int(data):
        for col in data.columns:
            for i, item in enumerate(data[col]):
                if isinstance(item, str):
                    try:
                        data.at[i, col] = float(item)
                    except ValueError:
                        if data[col].dtype == 'object':
                            labels, _ = pd.factorize(data[col])
                            data[col] = labels
        return data

    df = str_to_int(df)

    # Remove rows with empty strings
    def remove_rows_with_empty_strings(df):
        mask = df.apply(lambda x: x == ' ')
        rows_with_empty_strings = mask.any(axis=1)
        df_cleaned = df[~rows_with_empty_strings]
        return df_cleaned

    df = remove_rows_with_empty_strings(df)
    dfn = df.sample(n=1300, random_state=42)

    # Split the data into features and target variable
    X = dfn.drop(["customerID", "Churn"], axis=1)
    y = dfn["Churn"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy * 100:.2f}%')

    return accuracy