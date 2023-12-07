import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataVisualization:
    """A class for visualizing and analyzing traffic data."""

    def __init__(self, file_path):
        """Initialize the DataVisualization class with a file path for the dataset."""

        self.file_path = file_path
        self.data = None

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset, including handling of dates and filtering by year."""

        self.data = pd.read_csv(self.file_path, dtype={"summons_number": str})
        self.data["issue_date"] = pd.to_datetime(
            self.data["issue_date"], errors="coerce"
        )
        current_year = pd.Timestamp.now().year
        self.data = self.data[
            (self.data["issue_date"].dt.year > 2000)
            & (self.data["issue_date"].dt.year <= current_year)
        ]
        self.data["issue_year"] = self.data["issue_date"].dt.year

    def clean_data(self):
        """Remove outlier records from the dataset based on the 99th percentile of fine amounts."""
        percentile_99 = self.data["fine_amount"].quantile(0.99)
        self.data_filtered = self.data[self.data["fine_amount"] <= percentile_99]

    def plot_fine_amount_distribution(self):
        """Plot the distribution of fine amounts with outliers removed."""
        sns.histplot(self.data_filtered["fine_amount"], kde=True)
        plt.title("Distribution of Fine Amount (with outliers removed)")
        plt.xlabel("Fine Amount")
        plt.ylabel("Frequency")
        plt.show()

    def plot_average_fine_over_years(self):
        """Plot the average fine amount over different years."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x="issue_year", y="fine_amount", data=self.data_filtered, estimator="mean"
        )
        plt.title("Average Fine Amount Over Years (Cleaned Data)")
        plt.xlabel("Year")
        plt.ylabel("Average Fine Amount")
        plt.show()

    def display_data_overview(self):
        """Display an overview of the data, including head, description, and null value summary."""
        print(self.data.head())
        print(self.data.describe())
        print(self.data.info())
        print(self.data.isnull().sum())

    def handle_missing_values(self):
        """Handle missing values in the dataset, including filling and dropping as appropriate."""
        high_missing_cols = ["violation_status", "judgment_entry_date"]
        self.data = self.data.drop(columns=high_missing_cols)

        for col in ["violation_time", "violation", "county", "issuing_agency"]:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

        numerical_cols = [
            "fine_amount",
            "penalty_amount",
            "payment_amount",
            "amount_due",
            "precinct",
        ]
        for col in numerical_cols:
            self.data[col] = self.data[col].fillna(self.data[col].median())

        print("Missing values after handling:")
        print(self.data.isnull().sum())

    def plot_issuing_agencies_distribution(self):
        """Plot the distribution of issuing agencies based on the count of their appearances."""
        plt.figure(figsize=(10, 6))
        sns.countplot(
            y="issuing_agency",
            data=self.data,
            order=self.data["issuing_agency"].value_counts().index,
        )
        plt.title("Distribution of Issuing Agencies")
        plt.xlabel("Count")
        plt.ylabel("Issuing Agency")
        plt.show()

    def plot_top_violations(self):
        """Plot the top 10 most common violations."""
        top_violations = self.data["violation"].value_counts().nlargest(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(y=top_violations.index, x=top_violations.values)
        plt.title("Top 10 Violations")
        plt.xlabel("Count")
        plt.ylabel("Violation Type")
        plt.show()

    def plot_numerical_distributions(self, numerical_columns):
        """Plot the distributions of specified numerical columns."""
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(numerical_columns, 1):
            plt.subplot(2, 2, i)
            sns.histplot(self.data[column], kde=False, bins=30)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_fine_amount_by_issuing_agency(self):
        """Plot the distribution of fine amounts by different issuing agencies."""
        plt.figure(figsize=(15, 8))
        sns.boxplot(
            x="fine_amount", y="issuing_agency", data=self.data, showfliers=False
        )
        plt.title("Fine Amount by Issuing Agency")
        plt.xlabel("Fine Amount")
        plt.ylabel("Issuing Agency")
        plt.show()

    def transform_and_handle_violation_time(self):
        """Transform and handle 'violation_time' column, including handling null values."""
        print(
            "Unique 'violation_time' values before transformation:",
            self.data["violation_time"].unique(),
        )
        self.data["violation_time"] = (
            self.data["violation_time"].str.replace("A", "AM").str.replace("P", "PM")
        )
        self.data["hour"] = pd.to_datetime(
            self.data["violation_time"], format="%I:%M%p", errors="coerce"
        ).dt.hour
        null_hours = self.data["hour"].isnull().sum()
        print(f"Number of null values in 'hour' after conversion: {null_hours}")
        self.data["hour"] = self.data["hour"].fillna(-1)

    def plot_violations_by_hour(self):
        """Plot the distribution of violations by the hour of the day."""
        violations_by_hour = self.data["hour"].value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=violations_by_hour.index, y=violations_by_hour.values)
        plt.title("Violations by Hour of the Day")
        plt.xlabel("Hour of the Day (-1 for Unknown)")
        plt.ylabel("Number of Violations")
        plt.xticks(rotation=90)
        plt.show()

    def transform_issue_date_and_violation_time(self):
        """Transform 'issue_date' and 'violation_time' columns to extract meaningful data."""
        print(
            "Unique 'issue_date' values before transformation:",
            self.data["issue_date"].unique(),
        )
        print(
            "Null 'issue_date' values before transformation:",
            self.data["issue_date"].isnull().sum(),
        )

        self.data["issue_date"] = pd.to_datetime(
            self.data["issue_date"], errors="coerce"
        )
        self.data["day_of_week"] = self.data["issue_date"].dt.dayofweek
        self.data["month"] = self.data["issue_date"].dt.month

        self.data["violation_time"] = (
            self.data["violation_time"].str.replace("A", "AM").str.replace("P", "PM")
        )
        self.data["hour"] = pd.to_datetime(
            self.data["violation_time"], format="%I:%M%p", errors="coerce"
        ).dt.hour
        self.data["hour"] = self.data["hour"].fillna(-1)

        print(
            "Null 'issue_date' values after transformation:",
            self.data["issue_date"].isnull().sum(),
        )

    def plot_violations_by_time(self):
        """Plot the distribution of violations by day of the week and month."""
        violations_by_day = self.data["day_of_week"].value_counts().sort_index()
        violations_by_month = self.data["month"].value_counts().sort_index()
        violations_by_hour = self.data["hour"].value_counts().sort_index()

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        sns.barplot(x=violations_by_day.index, y=violations_by_day.values)
        plt.title("Violations by Day of the Week")
        plt.xlabel("Day of the Week")
        plt.ylabel("Number of Violations")

        plt.subplot(1, 3, 2)
        sns.barplot(x=violations_by_month.index, y=violations_by_month.values)
        plt.title("Violations by Month")
        plt.xlabel("Month")
        plt.ylabel("Number of Violations")

        plt.tight_layout()
        plt.show()
