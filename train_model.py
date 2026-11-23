import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

from main import load_data, clean_data

MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

def prepare_features(data):
    data = data.copy()

    # Multi-label encoding: one campaign can use more than one channel.
    channel_lists = data["Channel_Used"].fillna("Unknown").apply(
        lambda value: [item.strip() for item in str(value).split(",") if item.strip()]
    )
    encoder = MultiLabelBinarizer()
    channel_values = encoder.fit_transform(channel_lists)
    channel_columns = ["Channel_" + str(name).replace(" ", "_") for name in encoder.classes_]
    channel_df = pd.DataFrame(channel_values, columns=channel_columns, index=data.index)

    data = pd.concat([data, channel_df], axis=1)
    data["Campaign_Month"] = data["Campaign_Month"].fillna(0)
    data["Campaign_Year"] = data["Campaign_Year"].fillna(0)

    categorical = ["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]
    numeric = [
        "Duration", "Impressions", "Clicks", "Leads", "Conversions",
        "Acquisition_Cost", "Engagement_Score", "Campaign_Month", "Campaign_Year"
    ]
    feature_columns = categorical + numeric + channel_columns
    return data, feature_columns, categorical, numeric, channel_columns, encoder

def make_preprocessor(categorical, numeric):
    categorical_pipe = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"))
    ])
    numeric_pipe = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    return ColumnTransformer(transformers=[
        ("category", categorical_pipe, categorical),
        ("numbers", numeric_pipe, numeric)
    ])

def train():
    data = clean_data(load_data())
    data, features, categorical, numeric, channel_columns, channel_encoder = prepare_features(data)
    preprocessor = make_preprocessor(categorical, numeric + channel_columns)

    # ---------------- REVENUE REGRESSION ----------------
    # ROI is deliberately not used because it can be calculated from revenue.
    X_revenue = data[features]
    y_revenue = data["Revenue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_revenue, y_revenue, test_size=0.2, random_state=42
    )

    revenue_model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    revenue_model.fit(X_train, y_train)
    revenue_predictions = revenue_model.predict(X_test)

    mae = mean_absolute_error(y_test, revenue_predictions)
    rmse = mean_squared_error(y_test, revenue_predictions) ** 0.5
    r2 = r2_score(y_test, revenue_predictions)

    print("\nREVENUE PREDICTION RESULTS")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))
    joblib.dump(revenue_model, f"{MODEL_FOLDER}/revenue_model.pkl")

    # ---------------- PROFIT / LOSS CLASSIFICATION ----------------
    # ROI and Revenue are excluded to avoid giving the answer to the model.
    classification_features = [
        col for col in features
        if col not in ["Revenue", "ROI"]
    ]
    X_profit = data[classification_features]
    y_profit = data["Profit_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_profit, y_profit, test_size=0.2, random_state=42, stratify=y_profit
    )

    profit_preprocessor = make_preprocessor(categorical, numeric + channel_columns)
    profit_model = Pipeline(steps=[
        ("preprocessing", profit_preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    profit_model.fit(X_train, y_train)
    profit_predictions = profit_model.predict(X_test)

    print("\nPROFIT / LOSS RESULTS")
    print("Accuracy:", round(accuracy_score(y_test, profit_predictions), 4))
    print("Precision:", round(precision_score(y_test, profit_predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, profit_predictions, zero_division=0), 4))
    print("F1 Score:", round(f1_score(y_test, profit_predictions, zero_division=0), 4))
    joblib.dump(profit_model, f"{MODEL_FOLDER}/profit_model.pkl")

    # Save the input feature information for the Streamlit app.
    metadata = {
        "categorical": categorical,
        "numeric": numeric,
        "channel_columns": channel_columns,
        "classification_features": classification_features,
        "all_features": features
    }
    joblib.dump(metadata, f"{MODEL_FOLDER}/metadata.pkl")
    joblib.dump(channel_encoder, f"{MODEL_FOLDER}/channel_encoder.pkl")
    print("\nModels saved in the models folder.")

if __name__ == "__main__":
    train()
