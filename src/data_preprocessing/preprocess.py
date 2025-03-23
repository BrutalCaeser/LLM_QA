import pandas as pd
import os

def debug_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up TWO levels to reach project root: D:\Projects 2025 ML\LLM_QA
    project_root = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(project_root, "data", "raw", "hotel_bookings.csv")
    # Output path
    #output_path = os.path.join(project_root, "data", "processed", "cleaned_bookings.parquet")
    print(f"2. Correct CSV path: {csv_path}")  # Verify this path!
  # Debug paths
    print(f"INPUT PATH: {csv_path}")

    return csv_path
def clean_data():
    # Load data
    csv_path = debug_paths()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up TWO levels to reach project root: D:\Projects 2025 ML\LLM_QA
    project_root = os.path.dirname(os.path.dirname(script_dir))
    output_path = os.path.join(project_root, "data", "processed", "cleaned_bookings.parquet")
    df = pd.read_csv(csv_path)

    # Handle missing values
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(-1).astype(int)
    df['company'] = df['company'].fillna(-1).astype(int)

    # Standardize meal categories
    df['meal'] = df['meal'].replace('Undefined', 'SC')

    # Fix deposit type naming
    df['deposit_type'] = df['deposit_type'].replace('Non Refund', 'Non Refundable')

    # Fix negative ADR (impossible values)
    median_adr = df['adr'].median()
    df['adr'] = df['adr'].apply(lambda x: median_adr if x < 0 else x)

    # Cap extreme ADR at 99th percentile
    adr_cap = df['adr'].quantile(0.99)
    df['adr'] = df['adr'].apply(lambda x: adr_cap if x > adr_cap else x)

    # Remove invalid 0-adult bookings
    df = df[df['adults'] > 0]

    # Create unified arrival date
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'] + '-' +
        df['arrival_date_day_of_month'].astype(str),
        errors='coerce'
    )

    # Feature engineering
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    # Drop redundant columns
    df = df.drop(columns=['arrival_date_year', 'arrival_date_month',
                          'arrival_date_day_of_month', 'reservation_status_date'])
    # Save cleaned data
    df.to_parquet(output_path)
    return df
if __name__ == "__main__":
    # Path setup (PyCharm project root is the working directory)
    clean_data()
