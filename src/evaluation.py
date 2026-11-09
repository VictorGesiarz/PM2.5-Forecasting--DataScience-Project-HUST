"""Baselines, models, chronological cross-validation and Diebold-Mariano test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import AQI_BINS


# --------------------------------------------------------------------------- #
# Baselines (sklearn-compatible so they go through the same CV loop)
# --------------------------------------------------------------------------- #
class Persistence(BaseEstimator, RegressorMixin):
    """Tomorrow = today."""

    def __init__(self, col: str = "pm25_lag0"):
        self.col = col

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[self.col].to_numpy()


class SeasonalMean(BaseEstimator, RegressorMixin):
    """Mean of the target month in the training set (climatology)."""

    def fit(self, X, y):
        target_month = (X.index + pd.Timedelta(days=1)).month
        self.means_ = pd.Series(np.asarray(y), index=target_month).groupby(level=0).mean()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, X):
        target_month = (X.index + pd.Timedelta(days=1)).month
        return np.array([self.means_.get(m, self.global_) for m in target_month])


def make_models(random_state: int = 0) -> dict:
    return {
        "Persistence": Persistence(),
        "Seasonal mean": SeasonalMean(),
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            RidgeCV(alphas=np.logspace(-2, 3, 30)),
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, max_features=0.5,
            n_jobs=-1, random_state=random_state,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.03, max_iter=400, max_leaf_nodes=15, min_samples_leaf=10,
            l2_regularization=1.0, random_state=random_state,
        ),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def aqi_index(x) -> np.ndarray:
    return np.digitize(np.asarray(x, dtype=float), AQI_BINS[1:-1], right=True)


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "AQI_acc": float(np.mean(aqi_index(y_true) == aqi_index(y_pred))),
    }


# --------------------------------------------------------------------------- #
# Chronological cross-validation
# --------------------------------------------------------------------------- #
def expanding_folds(dates: pd.DatetimeIndex, n_folds: int = 5, min_train_days: int = 365,
                    gap_days: int = 1):
    """Expanding-window folds on calendar time (not row counts).

    The period after `min_train_days` is cut into `n_folds` consecutive test
    blocks; each fold trains on everything strictly before its block minus a
    `gap_days` gap (target of day t is day t+1, so a 1-day gap prevents the
    last training target from being the first test day's input).
    """
    start = dates.min() + pd.Timedelta(days=min_train_days)
    edges = pd.date_range(start, dates.max() + pd.Timedelta(days=1), periods=n_folds + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        train = dates < a - pd.Timedelta(days=gap_days)
        test = (dates >= a) & (dates < b)
        if test.sum() and train.sum():
            yield np.where(train)[0], np.where(test)[0]


def cross_validate(models: dict, X: pd.DataFrame, y: pd.Series, folds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-fold metrics and out-of-fold predictions for every model."""
    rows, preds = [], {}
    folds = list(folds)
    for name, model in models.items():
        p = pd.Series(np.nan, index=X.index)
        for k, (tr, te) in enumerate(folds):
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            p.iloc[te] = m.predict(X.iloc[te])
            rows.append({"model": name, "fold": k, "n_test": len(te),
                         "test_start": X.index[te[0]].date(), **metrics(y.iloc[te], p.iloc[te])})
        preds[name] = p
    return pd.DataFrame(rows), pd.DataFrame(preds)


# --------------------------------------------------------------------------- #
# Diebold-Mariano test (Harvey-Leybourne-Newbold small-sample correction)
# --------------------------------------------------------------------------- #
def diebold_mariano(y, pred_model, pred_ref, h: int = 1, loss: str = "abs") -> dict:
    """H0: equal accuracy. Negative statistic = model better than reference.

    For h=1 the loss differential is assumed serially uncorrelated beyond lag
    h-1; the HAC variance uses lags 0..h-1. Returns stat and two-sided p-value
    (Student t with n-1 df, HLN correction).
    """
    y, a, b = map(np.asarray, (y, pred_model, pred_ref))
    e1, e2 = y - a, y - b
    d = np.abs(e1) - np.abs(e2) if loss == "abs" else e1**2 - e2**2
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"DM": stat, "p_value": p, "mean_loss_diff": dbar, "n": n}
