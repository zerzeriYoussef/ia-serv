import pandas as pd
from app.services.data.scan_service import ScanService
from app.services.data.cleaning_service import CleaningService
import json

def main():
    # 1. Create a sample DataFrame with duplicates
    # Notice that row 0 and row 2 have the same Name and Age, but different 'id'
    data = {
    "id": [
        101, 102, 103, 104, 105,
        106, 107, 108, 109, 110,
        111, 112, 113, 114, 115,
        116, 117, 118, 119, 120
    ],
    "Name": [
        "Alice", "Bob", "Alice", "Charlie", "Diana",
        "Bob", "Eve", "Frank", "Grace", "Heidi",
        "Ivan", "Judy", "Mallory", "Oscar", "Peggy",
        "Trent", "Victor", "Walter", "Diana", "Grace"
    ],
    "Age": [
        25, 30, 25, 35, 28,
        30, 22, 40, 31, 29,
        45, 38, 33, 50, 27,
        44, 36, 41, 28, 31
    ],
    "City": [
        "Paris", "Lyon", "Paris", "Nice", "Rabat",
        "Lyon", "Casa", "Tunis", "Alger", "Paris",
        "Madrid", "Rome", "Berlin", "London", "Lisbon",
        "Dublin", "Brussels", "Vienna", "Rabat", "Alger"
    ],
    "Salary": [
        50000, 60000, 50000, 70000, 55000,
        60000, 42000, 80000, 62000, 58000,
        90000, 76000, 67000, 100000, 53000,
        88000, 72000, 81000, 55000, 62000
    ]
}

    df = pd.DataFrame(data)

    print("--- 1. Original DataFrame ---")
    print(df.to_string())
    print("\n")

    # 2. Test ScanService (What the frontend uses to display errors)
    print("--- 2. Testing ScanService (Duplicate Detection) ---")
    # Setting threshold high to avoid outlier flags interfering
    report = ScanService.scan_dataframe(df, outlier_threshold=10.0)
    
    dup_issues = [issue for issue in report['issues'] if issue['issue_type'] == 'duplicate']
    print(f"Found {len(dup_issues)} duplicate issues.")
    for issue in dup_issues:
        print(f"Row Index {issue['row_index']} -> {issue['description']}")
    print("\n")

    # 3. Test CleaningService (What actually removes the duplicates)
    print("--- 3. Testing CleaningService (Duplicate Removal) ---")
    profile = {
        "remove_duplicates": True, 
        "duplicate_keep": "best" # Keeps the best row
    }
    clean_df, clean_report = CleaningService.clean_dataframe(df, profile)

    print("Cleaned DataFrame (Row index 2 should be gone):")
    print(clean_df.to_string())
    
    dup_report = clean_report['changes'].get('duplicates', {})
    print(f"\nDuplicates found in report: {dup_report.get('duplicates_found')}")
    print(f"Duplicates removed in report: {dup_report.get('duplicates_removed')}")

if __name__ == "__main__":
    main()
