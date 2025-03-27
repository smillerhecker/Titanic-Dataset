# Titanic Survival Prediction Model

## 📌 Overview
This project aims to predict Titanic passengers' survival using a **Random Forest Regressor**. The model preprocesses data using **Pipelines and ColumnTransformer** to handle missing values and categorical encoding. However, the model currently has an accuracy issue with a **Mean Absolute Error (MAE) of 0.0**, which suggests possible preprocessing or target variable problems.

## 📂 Dataset
The dataset used in this project is stored in `tested.csv`. It includes information about Titanic passengers such as:
- `Pclass` (Ticket class)
- `Sex` (Gender, mapped to numerical values)
- `Age` (Passenger age)
- `SibSp` (Number of siblings/spouses aboard)
- `Parch` (Number of parents/children aboard)
- `Ticket` (Ticket number)
- `Fare` (Ticket price)
- `Cabin` (Cabin number)
- `Embarked` (Port of Embarkation)
- `Survived` (Target variable: 0 = No, 1 = Yes)

## 🛠️ Installation & Requirements
To run this project, install the following dependencies:
```bash
pip install pandas scikit-learn
```

## 🚀 Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/titanic-prediction.git
cd titanic-prediction
```

### 2️⃣ Run the Model
Make sure `tested.csv` is in the same directory, then execute:
```bash
python titanic_model.py
```

## 🔄 Data Preprocessing
1. **Handling Missing Values**
   - Numerical columns: Filled with a constant value.
   - Categorical columns: Filled with the most frequent value.
2. **Encoding Categorical Features**
   - `Sex` manually mapped (`male` → 0, `female` → 1).
   - Other categorical features encoded using **OneHotEncoder**.
3. **Train-Test Split**
   - 80% training, 20% testing.

## 📊 Model Training
- Uses `RandomForestRegressor` with 100 estimators and `random_state=0`.
- Trained on preprocessed data.

## 🏆 Model Evaluation
- Predicted values vs. actual values are compared using **Mean Absolute Error (MAE)**.
- **Current MAE: 0.0**, indicating an issue in data preprocessing or model selection.

## ⚠️ Troubleshooting Accuracy Issue
1. **Check Target Variable (`y`)**
   - Since `Survived` is a classification target (0 or 1), using `RandomForestRegressor` (for regression) instead of `RandomForestClassifier` is likely causing poor performance.
   - **Solution:** Use `RandomForestClassifier`:
     ```python
     from sklearn.ensemble import RandomForestClassifier
     model = RandomForestClassifier(n_estimators=100, random_state=0)
     ```
2. **Handle `Ticket` and `Cabin` Features**
   - These may not be useful as they contain too many unique values.
   - **Solution:** Drop them from `feature` selection.
3. **Check Data Imbalance**
   - If most values in `Survived` are 0s, the model might always predict 0.
   - **Solution:** Check class distribution:
     ```python
     print(df["Survived"].value_counts())
     ```

## 📜 License
This project is open-source under the **MIT License**.

---
Feel free to fork and improve! 🚀
