import pandas as pd
import os

uploads_dir = 'c:/stage/auth/IA-service/uploads'
for file in os.listdir(uploads_dir):
    if file.endswith('.csv'):
        try:
            head = pd.read_csv(os.path.join(uploads_dir, file), nrows=5)
            if 'price_per_cup' in head.columns and 'continent' in head.columns:
                print(file)
                break
        except Exception:
            pass
