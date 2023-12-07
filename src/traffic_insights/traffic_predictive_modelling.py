import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np


class PredictiveModel:
    """Class for handling various predictive modeling tasks."""

    def __init__(self, file_path):
        """Initialize the PredictiveModel with a file path for the dataset."""
        self.file_path = file_path
        self.data = None

    def load_and_preprocess(self):
        """Load the dataset from the file path and perform initial preprocessing steps."""
        # Load the dataset
        self.data = pd.read_csv(self.file_path)

        # Display initial information about the dataset
        print(self.data.head())
        print(self.data.describe())
        print(self.data.info())

        # Handle missing values
        high_missing_cols = ["violation_status", "judgment_entry_date"]
        self.data = self.data.drop(columns=high_missing_cols)

        # Fill missing values
        categorical_cols = ["violation_time", "violation", "county", "issuing_agency"]
        for col in categorical_cols:
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

        print(self.data.isnull().sum())
        return self.data

    def encode_categorical_columns(self, categorical_columns):
        """Encode specified categorical columns using one-hot encoding."""
        self.data_encoded = pd.get_dummies(self.data, columns=categorical_columns)
        print("Categorical columns one-hot encoded:")
        return self.data_encoded

    def split_dataset(
        self, drop_columns, target_column, test_size=0.2, random_state=42
    ):
        """Split the dataset into training and testing sets after dropping specified columns."""
        # Drop irrelevant columns
        data_model = self.data_encoded.drop(columns=drop_columns)

        # Separate features (X) and target variable (y)
        X = data_model.drop(columns=target_column)
        y = data_model[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def handle_non_numeric_columns(self, X_train, X_test, column_to_encode=None):
        """Handle non-numeric columns in the dataset, including optional encoding."""
        # Identify non-numeric columns
        non_numeric_columns = X_train.select_dtypes(include=["object"]).columns
        print(f"Non-numeric columns in X_train before handling: {non_numeric_columns}")

        # Encoding specified non-numeric column
        if column_to_encode in non_numeric_columns:
            X_train = pd.get_dummies(
                X_train, columns=[column_to_encode], drop_first=True
            )
            X_test = pd.get_dummies(X_test, columns=[column_to_encode], drop_first=True)

        # Drop any remaining non-numeric columns
        remaining_non_numeric = X_train.select_dtypes(include=["object"]).columns
        X_train = X_train.drop(columns=remaining_non_numeric)
        X_test = X_test.drop(columns=remaining_non_numeric)

        # Ensure both datasets have the same dummy columns
        X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

        print(
            f"Non-numeric columns in X_train after handling: {X_train.select_dtypes(include=['object']).columns}"
        )
        return X_train, X_test

    def recombine_encode_scale(self, X_train, X_test, column_to_encode=None):
        """Recombine training and testing sets, encode and scale features, then split them back."""
        # Recombine X_train and X_test
        X_combined = pd.concat([X_train, X_test], axis=0)

        # Perform one-hot encoding if specified column exists
        if column_to_encode and column_to_encode in X_combined.columns:
            X_combined = pd.get_dummies(X_combined, columns=[column_to_encode])

        # Drop any remaining non-numeric columns
        non_numeric_columns = X_combined.select_dtypes(include=["object"]).columns
        X_combined = X_combined.drop(columns=non_numeric_columns)

        # Separate them back into train and test sets
        X_train = X_combined.iloc[: len(X_train), :]
        X_test = X_combined.iloc[len(X_train) :, :]

        # Initialize and apply StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert scaled arrays back into DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        return X_train_scaled, X_test_scaled

    def recombine_encode_scale(self, X_train, X_test, column_to_encode=None):
        """Recombine training and testing sets, encode and scale features, then split them back."""
        # Recombine X_train and X_test
        X_combined = pd.concat([X_train, X_test], axis=0)

        # Fill NaN values with the median of each column
        X_combined = X_combined.fillna(X_combined.median())

        # Perform one-hot encoding if specified column exists
        if column_to_encode and column_to_encode in X_combined.columns:
            X_combined = pd.get_dummies(X_combined, columns=[column_to_encode])

        # Separate them back into train and test sets
        X_train = X_combined.iloc[: len(X_train), :]
        X_test = X_combined.iloc[len(X_train) :, :]

        # Initialize and apply StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert scaled arrays back into DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        return X_train_scaled, X_test_scaled

    def train_and_evaluate(self, X_train_scaled, y_train, X_test_scaled, y_test, model):
        """Train a given model and evaluate it using Mean Squared Error."""
        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model using Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        model_name = type(model).__name__
        print(f"{model_name} Mean Squared Error: {mse}")
        return mse

    def create_pipeline_and_evaluate(self, X, y, numerical_cols, categorical_cols):
        """Create a pipeline with preprocessing and a model, and evaluate it using cross-validation."""
        # Transformers for imputation and scaling/encoding
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Preprocessor for column transformation
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Complete pipeline with preprocessing and RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # Perform cross-validation
        cross_val_scores = cross_val_score(
            pipeline, X, y, cv=5, scoring="neg_mean_squared_error"
        )

        # Calculate RMSE
        rmse_scores = np.sqrt(-cross_val_scores)
        print(f"Cross-validated RMSE scores: {rmse_scores}")
        print(f"Mean RMSE: {np.mean(rmse_scores)}")
