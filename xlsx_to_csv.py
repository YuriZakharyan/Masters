import pandas as pd

def convert_xlsx_to_csv(xlsx_path, csv_path, sheet_name=0):
    """
    Converts an Excel (.xlsx) file to CSV.

    Parameters:
        xlsx_path (str): Path to the input .xlsx file.
        csv_path (str): Path to save the output .csv file.
        sheet_name (str or int): Sheet name or index to convert (default is the first sheet).
    """
    try:
        # Read the Excel file
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        
        # Write to CSV
        df.to_csv(csv_path, index=False)
        print(f"Successfully converted '{xlsx_path}' to '{csv_path}'")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
convert_xlsx_to_csv("daily_FOREX_AMD_sentiment_data_all.xlsx", "daily_FOREX_AMD_sentiment_data_all.csv")
convert_xlsx_to_csv("daily_FOREX_USD_sentiment_data_all.xlsx", "daily_FOREX_USD_sentiment_data_all.csv")

