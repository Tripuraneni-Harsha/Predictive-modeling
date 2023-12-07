# data_exploration.py
import pandas as pd


class DataAnalyzer:
    """
    A class used to perform basic data exploration tasks.

    Attributes
    ----------
    file_path : str
        The file path to the dataset.
    data : DataFrame
        The pandas DataFrame holding the loaded data.

    Methods
    -------
    load_data():
        Loads data from the specified file path into a pandas DataFrame.
    display_head():
        Returns the first few rows of the DataFrame.
    get_info():
        Prints information about the DataFrame including the data type of each column and non-null values.
    describe_data():
        Returns summary statistics of the DataFrame's numerical columns.
    check_missing_values():
        Returns a series indicating the count of missing values in each column.
    check_unique_values():
        Returns a series indicating the count of unique values in each column.
    """

    def __init__(self, file_path):
        """
        Constructs all the necessary attributes for the DataAnalyzer object.

        Parameters
        ----------
        file_path : str
            The file path to the dataset.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Loads data from the specified file path into a pandas DataFrame.

        Returns
        -------
        DataFrame
            The loaded data.
        """
        self.data = pd.read_csv(self.file_path)
        return self.data

    def display_head(self):
        """
        Returns the first few rows of the DataFrame.

        Returns
        -------
        DataFrame
            The first few rows of the DataFrame.
        """
        return self.data.head()

    def get_info(self):
        """
        Prints information about the DataFrame including the data type of each column and non-null values.

        Returns
        -------
        None
        """
        return self.data.info()

    def describe_data(self):
        """
        Returns summary statistics of the DataFrame's numerical columns.

        Returns
        -------
        DataFrame
            Summary statistics of the DataFrame's numerical columns.
        """
        return self.data.describe()

    def check_missing_values(self):
        """
        Returns a series indicating the count of missing values in each column.

        Returns
        -------
        Series
            The count of missing values in each column.
        """
        return self.data.isnull().sum()

    def check_unique_values(self):
        """
        Returns a series indicating the count of unique values in each column.

        Returns
        -------
        Series
            The count of unique values in each column.
        """
        return self.data.nunique()
