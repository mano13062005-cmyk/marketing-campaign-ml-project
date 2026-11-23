import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_FOLDER = "data"
FILES = {
    "Tira": "tira_campaign_data_with_nulls.csv",
    "Purplle": "purplle_campaign_data_with_nulls.csv",
    "Nykaa": "nykaa_campaign_data_with_nulls.csv",
}

def load_data():
    """Load the three brand CSV files and combine them into one table."""
    frames = []
    for brand, filename in FILES.items():
        path = os.path.join(DATA_FOLDER, filename)
        df = pd.read_csv(path)
        df["Brand"] = brand
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    return data

def clean_data(data):
    """Basic cleaning: remove duplicates and fill missing values."""
    data = data.copy()
    data = data.drop_duplicates()

    # Convert number columns to numeric values.
    number_columns = [
        "Duration", "Impressions", "Clicks", "Leads", "Conversions",
        "Revenue", "Acquisition_Cost", "ROI", "Engagement_Score"
    ]
    for column in number_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].median())

    # Fill missing text values with the most common value.
    text_columns = [
        "Campaign_Type", "Target_Audience", "Channel_Used",
        "Language", "Customer_Segment"
    ]
    for column in text_columns:
        data[column] = data[column].fillna("Unknown").astype(str)

    # Convert the date column and create simple date features.
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data["Campaign_Month"] = data["Date"].dt.month.fillna(0).astype(int)
    data["Campaign_Year"] = data["Date"].dt.year.fillna(0).astype(int)

    # Profit flag: ROI greater than zero means profit.
    data["Profit_Flag"] = (data["ROI"] > 0).astype(int)
    return data

def show_summary(data):
    print("\nDATASET SHAPE:", data.shape)
    print("\nFIRST FIVE ROWS:")
    print(data.head())
    print("\nMISSING VALUES AFTER CLEANING:")
    print(data.isnull().sum())
    print("\nBRAND-WISE CAMPAIGNS:")
    print(data["Brand"].value_counts())

def save_chart(data):
    brand_revenue = data.groupby("Brand")["Revenue"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    brand_revenue.plot(kind="bar")
    plt.title("Average Campaign Revenue by Brand")
    plt.xlabel("Brand")
    plt.ylabel("Average Revenue")
    plt.tight_layout()
    plt.savefig("brand_revenue.png")
    print("\nChart saved as brand_revenue.png")

if __name__ == "__main__":
    raw_data = load_data()
    cleaned_data = clean_data(raw_data)
    cleaned_data.to_csv("cleaned_campaign_data.csv", index=False)
    show_summary(cleaned_data)
    save_chart(cleaned_data)
    print("\nCleaning completed. File saved: cleaned_campaign_data.csv")
