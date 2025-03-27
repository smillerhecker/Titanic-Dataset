# Titanic Survival Prediction Model

## 📌 Overview
This project aims to predict Titanic passengers' survival using a **Random Forest Classifier**. The model preprocesses data using **Pipelines and ColumnTransformer** to handle missing values and categorical encoding. The model has been successfully troubleshooted, and the accuracy has improved.

## 📂 Dataset
The dataset used in this project is stored in `tested.csv`. It includes information about Titanic passengers such as:
- `Pclass` (Ticket class)
- `Sex` (Gender, mapped to numerical values)
- `Age` (Passenger age)
- `SibSp` (Number of siblings/spouses aboard)
- `Parch` (Number of parents/children aboard)
- `Fare` (Ticket price)
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
3. **Dropped Irrelevant Features**
   - `Ticket` and `Cabin` were removed due to their high uniqueness and limited predictive power.
4. **Train-Test Split**
   - 80% training, 20% testing.

## 📊 Model Training
- Uses `RandomForestClassifier` with 100 estimators and `random_state=0`.
- Trained on preprocessed data.

## 🏆 Model Evaluation
- Predicted values vs. actual values are compared using **Mean Absolute Error (MAE)**.
- **Current MAE: 0.0**, meaning the model is predicting accurately on the given test set.

## 📜 License
This project is open-source under the **MIT License**.

---
Feel free to fork and improve! 🚀
