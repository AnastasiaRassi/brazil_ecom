import pandas as pd

class PreprocessingUtils:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def rows_after_checks(self, checks):
        """
        Applies a series of checks to the DataFrame and returns the number of rows after each check.
        Args:
            checks (dict): A dictionary where keys are check names and values are functions that take a DataFrame and return a boolean mask.
        Returns:
            difference in row count before and after each check.
        """
        for name, rule in checks.items():
            # Print the name of the check
            print(f"\nChecking: {name}")
            before = df.shape[0]
            mask = rule(df)
            df = df[~mask]
            print("Rows after:", df.shape[0] - before)
        