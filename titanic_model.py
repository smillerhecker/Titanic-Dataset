# importing neccessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Read the dataset
path = "/kaggle/input/test-file/tested.csv"
df = pd.read_csv(path)

# Preprocessing
df["Sex"] = df["Sex"].map({'male':0,'female':1})

print(df.shape)
df.head()

# Defining features and target
feature = ["Pclass","Sex","Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
X = df[feature].copy()
y = df["Survived"]

# Select categorical columns
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 
                        X[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

print(numerical_cols)
print(categorical_cols)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

encoded = preprocessor.fit_transform(X)
encoded = pd.DataFrame(encoded)
encoded.head()

# Splitting datset into training and testing
X_train,X_test ,y_train , y_test = train_test_split(encoded, y, test_size=0.2)

# Defining out ML model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Evaluation of the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_pred, y_test)
print(mae)
print("y_pred",y_pred[:5])
print("y_test",y_test[:5])