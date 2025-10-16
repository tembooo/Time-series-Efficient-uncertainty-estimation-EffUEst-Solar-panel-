"""
Module containing helper functions tailored for the Transfep dataset
"""

import numpy as np
import polars as pl

type NormParams = dict[str, tuple[float, float]]

STARTING_NORM_PARAMS: NormParams = {
    "cloud_amount": (0., 9.),
    "horizontal_visibility": (0., 5.e4),
    "relative_humidity": (0., 100.),
}

SOLAR_FEATURES = [
    "diffuse_r",
    "full_solar",
    "global_r",
    "sunshine"
]

UNNORMALIZED_FEATURES = ["elspot", "energy"]

ROOT: str = "../../data/"
RNG: np.random.Generator = np.random.default_rng()
REPS: int = 10


def transform_wind_features(df: pl.DataFrame) -> pl.DataFrame:
  """
  Returns a Dataframe with cartesian wind columns instead of the original polar
  representation.
  """
  return df.with_columns(
      (
          pl.col("wind_speed") * (2 * np.pi *
                                  pl.col("wind_direction") / 360).sin()
      ).alias("wind_sin"),
      (
          pl.col("wind_speed") * (2 * np.pi *
                                  pl.col("wind_direction") / 360).cos()
      ).alias("wind_cos")
  ).select(
      pl.all().exclude([
          "wind_speed", "wind_direction"
      ])
  )


def add_datetime_features(df: pl.DataFrame) -> pl.DataFrame:
  """
  Returns a Dataframe with additional datetime cyclical features.
  """
  return df.with_columns(
      (
          pl.col("datetime").dt.hour() * 2 * np.pi / 24
      ).sin().alias("hour_sin"),
      (
          pl.col("datetime").dt.hour() * 2 * np.pi / 24
      ).cos().alias("hour_cos"),
      (
          pl.col("datetime").dt.weekday() * 2 * np.pi / 7
      ).sin().alias("dow_sin"),
      (
          pl.col("datetime").dt.weekday() * 2 * np.pi / 7
      ).cos().alias("dow_cos"),
      (
          pl.col("datetime").dt.ordinal_day() * 2 * np.pi /
          pl.when(pl.col("datetime").dt.is_leap_year()).then(
              366
          ).otherwise(
              365
          )
      ).sin().alias("doy_sin"),
      (
          pl.col("datetime").dt.ordinal_day() * 2 * np.pi /
          pl.when(pl.col("datetime").dt.is_leap_year()).then(
              366
          ).otherwise(
              365
          )
      ).cos().alias("doy_cos")
  )


def get_dataset_n(
    n: int
) -> pl.DataFrame:
  """
  Returns dataset 'n' according to the filename as a Polars Dataframe.
  """
  return pl.read_parquet(
      source=ROOT + f"dataset_{n}.parquet"
  ).with_columns(  # Remove wrong readings
      pl.all().exclude("^.*_temperature$").clip(lower_bound=0)
  )


def get_curated_datasets(
    relevant_variables: list[str] = None,
    log_solar: bool = False
) -> tuple[pl.DataFrame, ...]:
  """
  Returns dataframes for train/test/validation (2, 1, 0) containing only the
  relevant columns or all columns by default.
  """
  if relevant_variables is not None and "datetime" not in relevant_variables:
    relevant_variables.append("datetime")
  return tuple(
      [
          add_datetime_features(
              transform_wind_features(
                  get_dataset_n(n).with_columns(
                      full_solar=pl.col(
                          "full_solar"
                      ).log1p() if log_solar else pl.col("full_solar")
                  )
              )
          ).select(
              relevant_variables or pl.all()
          )
          for n in [2, 1, 0]
      ]
  )


def set_partial_placeholder(
    x_enc: np.ndarray,
    x_dec: np.ndarray,
    shift: int
) -> np.ndarray:
  x_mod = x_dec.copy()
  x_mod[..., shift:] = np.mean(
      np.concatenate(
          [
              x_enc,
              x_dec[..., :shift]
          ],
          -1
      ),
      -1, keepdims=True
  )

  return x_mod
