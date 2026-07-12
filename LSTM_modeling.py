from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# CONFIGURATION

DATE_COLUMN = "date"
TARGET_COLUMN = "AOT_t"
FEATURE_COLUMNS = ["Tmax", "Tmin", "Pr", "AOT", "wind"]

SEQUENCE_LENGTH = 10
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20

EPOCHS = 2000
BATCH_SIZE = 32
PATIENCE = 150
RANDOM_SEED = 42

HISTORICAL_DIR = Path("data/historical")
FUTURE_DIRS = {
    "ssp245": Path("data/future/ssp245"),
    "ssp585": Path("data/future/ssp585"),
}
OUTPUT_DIR = Path("results")

def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def city_identifier(path: Path) -> str:
    stem = re.sub(r"\s*\(\d+\)\s*$", "", path.stem.strip())
    stem = re.sub(r"(?i)([_-](new|historical|history|his))$", "", stem)
    return stem


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeError(f"Unable to decode {path.name}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def prepare_historical_data(path: Path) -> pd.DataFrame:
    data = load_table(path)

    required_columns = [
        DATE_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{path.name} is missing required columns: {missing_columns}"
        )

    data = data[required_columns].copy()
    data[DATE_COLUMN] = pd.to_datetime(
        data[DATE_COLUMN],
        errors="coerce",
    )

    for column in [TARGET_COLUMN, *FEATURE_COLUMNS]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = (
        data.dropna(subset=[DATE_COLUMN, TARGET_COLUMN])
        .sort_values(DATE_COLUMN)
        .drop_duplicates(subset=[DATE_COLUMN], keep="last")
        .reset_index(drop=True)
    )

    if len(data) <= 2 * SEQUENCE_LENGTH:
        raise ValueError(
            f"{path.name} does not contain enough observations."
        )

    return data


def prepare_future_data(path: Path) -> pd.DataFrame:
    data = load_table(path)

    required_columns = [
        DATE_COLUMN,
        *[column for column in FEATURE_COLUMNS if column != "AOT"],
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{path.name} is missing required columns: {missing_columns}"
        )

    data = data.copy()
    data[DATE_COLUMN] = pd.to_datetime(
        data[DATE_COLUMN],
        errors="coerce",
    )

    for column in required_columns:
        if column != DATE_COLUMN:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    return (
        data.dropna(subset=[DATE_COLUMN])
        .sort_values(DATE_COLUMN)
        .drop_duplicates(subset=[DATE_COLUMN], keep="last")
        .reset_index(drop=True)
    )

def create_sequences(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    dates: pd.Series,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    sequences = []
    targets = []
    target_dates = []

    for end_index in range(
        sequence_length - 1,
        len(feature_values) - 1,
    ):
        start_index = end_index - sequence_length + 1

        sequences.append(
            feature_values[start_index : end_index + 1]
        )
        targets.append(target_values[end_index])
        target_dates.append(dates.iloc[end_index + 1])

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        pd.DatetimeIndex(target_dates),
    )


def prepare_train_test_data(
    data: pd.DataFrame,
) -> dict[str, object]:
    split_index = int(len(data) * TRAIN_FRACTION)

    if split_index <= SEQUENCE_LENGTH:
        raise ValueError("The training period is too short.")

    training_data = data.iloc[:split_index].copy()
    testing_context = data.iloc[
        split_index - SEQUENCE_LENGTH :
    ].copy()

    feature_imputer = SimpleImputer(strategy="median")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    training_features_imputed = feature_imputer.fit_transform(
        training_data[FEATURE_COLUMNS]
    )
    training_features_scaled = feature_scaler.fit_transform(
        training_features_imputed
    )
    training_target_scaled = target_scaler.fit_transform(
        training_data[[TARGET_COLUMN]]
    )

    testing_features_imputed = feature_imputer.transform(
        testing_context[FEATURE_COLUMNS]
    )
    testing_features_scaled = feature_scaler.transform(
        testing_features_imputed
    )
    testing_target_scaled = target_scaler.transform(
        testing_context[[TARGET_COLUMN]]
    )

    x_train, y_train, _ = create_sequences(
        feature_values=training_features_scaled,
        target_values=training_target_scaled,
        dates=training_data[DATE_COLUMN],
        sequence_length=SEQUENCE_LENGTH,
    )

    x_test, y_test, test_dates = create_sequences(
        feature_values=testing_features_scaled,
        target_values=testing_target_scaled,
        dates=testing_context[DATE_COLUMN],
        sequence_length=SEQUENCE_LENGTH,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "test_dates": test_dates,
        "feature_imputer": feature_imputer,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }

def build_lstm_model(
    sequence_length: int,
    number_of_features: int,
) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(
                shape=(sequence_length, number_of_features)
            ),
            layers.LSTM(
                128,
                return_sequences=True,
            ),
            layers.Dropout(0.20),
            layers.LSTM(64),
            layers.Dropout(0.20),
            layers.Dense(
                32,
                activation="relu",
            ),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=[
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    return model


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> keras.callbacks.History:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        restore_best_weights=True,
    )

    return model.fit(
        x_train,
        y_train,
        validation_split=VALIDATION_FRACTION,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0,
    )


def evaluate_model(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_dates: pd.DatetimeIndex,
    target_scaler: StandardScaler,
) -> tuple[pd.DataFrame, dict[str, float]]:
    prediction_scaled = model.predict(
        x_test,
        verbose=0,
    )

    observed = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).ravel()

    predicted = target_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).ravel()

    prediction_table = pd.DataFrame(
        {
            DATE_COLUMN: test_dates,
            "observed_AOT": observed,
            "predicted_AOT": predicted,
        }
    )

    metrics = {
        "RMSE": float(
            np.sqrt(mean_squared_error(observed, predicted))
        ),
        "MAE": float(
            mean_absolute_error(observed, predicted)
        ),
        "R2": float(
            r2_score(observed, predicted)
        ),
    }

    return prediction_table, metrics

def find_matching_future_file(
    directory: Path,
    identifier: str,
) -> Path | None:
    if not directory.exists():
        return None

    normalized_identifier = normalize_identifier(identifier)
    candidates = sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
        ]
    )

    exact_matches = [
        path
        for path in candidates
        if normalize_identifier(path.stem) == normalized_identifier
    ]
    if exact_matches:
        return exact_matches[0]

    partial_matches = [
        path
        for path in candidates
        if normalized_identifier in normalize_identifier(path.stem)
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]

    return None


def forecast_future_aot(
    model: keras.Model,
    historical_data: pd.DataFrame,
    future_data: pd.DataFrame,
    feature_imputer: SimpleImputer,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> pd.DataFrame:
    rolling_window = historical_data[
        FEATURE_COLUMNS
    ].tail(SEQUENCE_LENGTH).copy()

    predictions = []

    for _, future_row in future_data.iterrows():
        window_imputed = feature_imputer.transform(
            rolling_window[FEATURE_COLUMNS]
        )
        window_scaled = feature_scaler.transform(
            window_imputed
        ).astype(np.float32)

        prediction_scaled = model.predict(
            window_scaled.reshape(
                1,
                SEQUENCE_LENGTH,
                len(FEATURE_COLUMNS),
            ),
            verbose=0,
        )

        predicted_aot = float(
            target_scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            )[0, 0]
        )

        predictions.append(predicted_aot)

        next_row = {
            feature: future_row.get(feature, np.nan)
            for feature in FEATURE_COLUMNS
        }
        next_row["AOT"] = predicted_aot

        rolling_window = pd.concat(
            [
                rolling_window.iloc[1:],
                pd.DataFrame([next_row]),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(
        {
            DATE_COLUMN: future_data[DATE_COLUMN].values,
            "predicted_AOT": predictions,
        }
    )

def run_workflow() -> None:
    set_reproducibility(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    historical_files = sorted(
        [
            path
            for path in HISTORICAL_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
        ]
    )

    metrics_records = []

    for historical_path in historical_files:
        identifier = city_identifier(historical_path)
        city_output_dir = OUTPUT_DIR / identifier
        city_output_dir.mkdir(parents=True, exist_ok=True)

        historical_data = prepare_historical_data(
            historical_path
        )
        prepared = prepare_train_test_data(
            historical_data
        )

        model = build_lstm_model(
            sequence_length=SEQUENCE_LENGTH,
            number_of_features=len(FEATURE_COLUMNS),
        )

        train_model(
            model=model,
            x_train=prepared["x_train"],
            y_train=prepared["y_train"],
        )

        prediction_table, metrics = evaluate_model(
            model=model,
            x_test=prepared["x_test"],
            y_test=prepared["y_test"],
            test_dates=prepared["test_dates"],
            target_scaler=prepared["target_scaler"],
        )

        prediction_table.to_csv(
            city_output_dir / "test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        model.save(
            city_output_dir / "lstm_model.keras"
        )

        metrics_records.append(
            {
                "city": identifier,
                "n_train_sequences": len(prepared["x_train"]),
                "n_test_sequences": len(prepared["x_test"]),
                **metrics,
            }
        )

        for scenario, scenario_directory in FUTURE_DIRS.items():
            future_path = find_matching_future_file(
                directory=scenario_directory,
                identifier=identifier,
            )

            if future_path is None:
                continue

            future_data = prepare_future_data(
                future_path
            )

            future_projection = forecast_future_aot(
                model=model,
                historical_data=historical_data,
                future_data=future_data,
                feature_imputer=prepared["feature_imputer"],
                feature_scaler=prepared["feature_scaler"],
                target_scaler=prepared["target_scaler"],
            )

            future_projection.to_csv(
                city_output_dir / f"{scenario}_projection.csv",
                index=False,
                encoding="utf-8-sig",
            )

    pd.DataFrame(metrics_records).to_csv(
        OUTPUT_DIR / "lstm_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )


run_workflow()
