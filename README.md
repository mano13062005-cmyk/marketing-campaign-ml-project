# Marketing Campaign Performance Prediction (Beginner Project)

This project combines the Nykaa, Purplle and Tira campaign CSV files and uses Python and Machine Learning to:
- Clean missing values and remove duplicate rows.
- Explore campaign performance by brand.
- Use multi-label encoding for campaigns that use multiple marketing channels.
- Predict campaign revenue (regression).
- Predict profit or loss (classification).
- Show predictions in a Streamlit web app.

## 1. Install Python

Install Python 3.10 or newer. During installation, select **Add Python to PATH**.

## 2. Open the project folder

Open Command Prompt or PowerShell inside this project folder.

## 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

## 4. Install the required libraries

```bash
pip install -r requirements.txt
```

## 5. Clean and inspect the data

```bash
python main.py
```

This creates:
- `cleaned_campaign_data.csv`
- `brand_revenue.png`

## 6. Train the Machine Learning models

```bash
python train_model.py
```

The trained models will be saved in the `models` folder:
- `revenue_model.pkl`
- `profit_model.pkl`
- `metadata.pkl`
- `channel_encoder.pkl`

The terminal displays MAE, RMSE, R², accuracy, precision, recall and F1-score.

## 7. Start the Streamlit app

```bash
streamlit run app.py
```

A browser tab opens. Enter campaign details and click **Predict Campaign Performance**.

## Project folder structure

```text
marketing_campaign_ml_beginner/
│
├── data/
│   ├── nykaa_campaign_data_with_nulls.csv
│   ├── purplle_campaign_data_with_nulls.csv
│   └── tira_campaign_data_with_nulls.csv
│
├── models/
├── main.py
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Important beginner explanations

- **Pandas:** reads and works with CSV data.
- **Data cleaning:** fills missing values and removes duplicate rows.
- **EDA:** explores patterns in campaign data.
- **Multi-label encoding:** converts multiple channel names into numeric columns.
- **Regression:** predicts a number, here campaign revenue.
- **Classification:** predicts a category, here profit or loss.
- **Streamlit:** turns the Python program into a simple web application.

## Important notes

- The profit label is created from `ROI > 0`, following the project brief.
- ROI is excluded from the profit/loss model so the answer is not directly given to the model.
- Revenue is excluded from the profit/loss model for the same reason.
- Revenue is excluded from the revenue model target inputs; ROI is also excluded because it may be calculated using revenue.
- The model scores depend on the dataset and should be reported as measured. Do not claim a target accuracy or R² unless your actual test results achieve it.
- Predictions are estimates, not guaranteed business outcomes.
