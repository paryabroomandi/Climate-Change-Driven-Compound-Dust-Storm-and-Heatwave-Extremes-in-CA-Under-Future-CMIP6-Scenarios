from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


# =============================================================================
# SETTINGS
# =============================================================================

TARGET = "AOT"
DATE_COLUMN = "date"
PREDICTORS = ["Pr", "Tmax", "Tmin", "Tave", "wind", "RH"]

TEST_FRACTION = 0.20
MINIMUM_ROWS = 60
RANDOM_STATE = 42
N_JOBS = 1
N_PERMUTATION_REPEATS = 30

RF_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "bootstrap": True,
    "criterion": "squared_error",
}

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
OUTPUT_FILENAME = "RF_feature_importance_by_city.csv"


# =============================================================================
# FUNCTIONS
# =============================================================================

def configure_logging() -> None:
    """Configure concise console messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def read_city_file(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel city file."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeError(f"Could not decode CSV file: {path.name}")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def get_city_name(path: Path, data: pd.DataFrame) -> str:
    """
    Obtain the city name from a city/station column when available;
    otherwise use the filename.

    Example:
        Abadan(4).csv -> Abadan
    """
    for column in ("city", "City", "CITY", "station", "Station"):
        if column in data.columns:
            values = data[column].dropna().astype(str).str.strip().unique()
            if len(values) == 1 and values[0]:
                return values[0]

    city = re.sub(r"\s*\(\d+\)\s*$", "", path.stem.strip())
    return city


def validate_data(data: pd.DataFrame, path: Path) -> None:
    """Check that all required columns are available."""
    required_columns = [DATE_COLUMN, TARGET, *PREDICTORS]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{path.name}: missing required columns {missing_columns}"
        )


def prepare_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, sort, and extract predictors and target."""
    selected = data[[DATE_COLUMN, TARGET, *PREDICTORS]].copy()

    selected[DATE_COLUMN] = pd.to_datetime(
        selected[DATE_COLUMN],
        errors="coerce",
    )

    for column in [TARGET, *PREDICTORS]:
        selected[column] = pd.to_numeric(
            selected[column],
            errors="coerce",
        )

    # Target values and dates cannot be imputed.
    selected = selected.dropna(subset=[DATE_COLUMN, TARGET])

    # Ensure a unique and chronological time series.
    selected = (
        selected.sort_values(DATE_COLUMN)
        .drop_duplicates(subset=[DATE_COLUMN], keep="last")
        .reset_index(drop=True)
    )

    if len(selected) < MINIMUM_ROWS:
        raise ValueError(
            f"Only {len(selected)} usable rows were found; "
            f"at least {MINIMUM_ROWS} are required."
        )

    entirely_missing = [
        feature
        for feature in PREDICTORS
        if selected[feature].notna().sum() == 0
    ]
    if entirely_missing:
        raise ValueError(
            f"Predictors contain no valid values: {entirely_missing}"
        )

    x = selected[PREDICTORS].copy()
    y = selected[TARGET].copy()

    return x, y


def temporal_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically to prevent future observations from entering
    the training period.
    """
    n_test = max(1, int(np.ceil(len(x) * TEST_FRACTION)))
    split_index = len(x) - n_test

    if split_index < 24:
        raise ValueError(
            f"Training period contains only {split_index} observations."
        )

    x_train = x.iloc[:split_index].copy()
    x_test = x.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return x_train, x_test, y_train, y_test


def build_model() -> Pipeline:
    """
    Create a reproducible pipeline.

    Predictor missing values are replaced using medians estimated only from
    the training period.
    """
    random_forest = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        **RF_PARAMETERS,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("random_forest", random_forest),
        ]
    )


def calculate_city_importance(path: Path) -> dict[str, float | str]:
    """Calculate normalized permutation importance for one city file."""
    raw_data = read_city_file(path)
    city = get_city_name(path, raw_data)

    validate_data(raw_data, path)
    x, y = prepare_data(raw_data)

    x_train, x_test, y_train, y_test = temporal_train_test_split(x, y)

    model = build_model()
    model.fit(x_train, y_train)

    permutation_result = permutation_importance(
        estimator=model,
        X=x_test,
        y=y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=N_PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    raw_importance = pd.Series(
        permutation_result.importances_mean,
        index=PREDICTORS,
        dtype=float,
    )

    # Negative values can occur because of test-sample variability.
    # They are set to zero before normalization.
    positive_importance = raw_importance.clip(lower=0.0)
    importance_sum = float(positive_importance.sum())

    if importance_sum > 0:
        normalized_importance = positive_importance / importance_sum
    else:
        normalized_importance = pd.Series(
            0.0,
            index=PREDICTORS,
            dtype=float,
        )

    return {
        "city": city,
        **{
            feature: float(normalized_importance[feature])
            for feature in PREDICTORS
        },
    }


def find_city_files(
    input_directory: Path,
    recursive: bool,
) -> list[Path]:
    """Find supported city files."""
    iterator = (
        input_directory.rglob("*")
        if recursive
        else input_directory.glob("*")
    )

    return sorted(
        path
        for path in iterator
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and path.name != OUTPUT_FILENAME
        )
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Random Forest permutation feature importance "
            "for AOT in multiple city files."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing city CSV or Excel files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the output CSV file.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search inside subdirectories.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the city-level feature-importance analysis."""
    configure_logging()
    args = parse_arguments()

    input_directory = args.input_dir.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()

    if not input_directory.is_dir():
        logging.error(
            "Input directory does not exist: %s",
            input_directory,
        )
        return 1

    city_files = find_city_files(
        input_directory=input_directory,
        recursive=args.recursive,
    )

    if not city_files:
        logging.error(
            "No supported city files were found in %s",
            input_directory,
        )
        return 1

    logging.info("Found %d city files.", len(city_files))

    results: list[dict[str, float | str]] = []

    for index, path in enumerate(city_files, start=1):
        logging.info(
            "[%d/%d] Processing %s",
            index,
            len(city_files),
            path.name,
        )

        try:
            results.append(calculate_city_importance(path))
        except Exception as error:
            logging.warning(
                "Skipped %s: %s",
                path.name,
                error,
            )

    if not results:
        logging.error("No city files were processed successfully.")
        return 1

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / OUTPUT_FILENAME

    result_table = pd.DataFrame(
        results,
        columns=["city", *PREDICTORS],
    )

    result_table = (
        result_table.sort_values("city")
        .reset_index(drop=True)
    )

    result_table.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    logging.info(
        "Saved feature importance for %d cities to %s",
        len(result_table),
        output_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
