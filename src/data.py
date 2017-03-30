import pandas as pd

def format_as_df(csv_file):
    df = pd.DataFrame.from_csv(csv_file, header=0)

    return df
