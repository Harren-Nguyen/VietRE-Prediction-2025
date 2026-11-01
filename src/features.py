"""
Import + clean dataset, address normalization, out-of-distribution check
Use for train.py and predict.py
"""

import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

TARGET = "price"
NUM_COLS = ["area", "floors", "bedrooms", "bathrooms"]
CAT_COLS = ["street", "ward", "province"]  # Extract from "address" feature in dataset
FEATURES = NUM_COLS + CAT_COLS
UNKNOWN = "unknown"

# Defined bounds (invalid = dropped)
VALID_RANGE = {"area": (5, 2500), "floors": (1, 12), "bedrooms": (1, 20), "bathrooms": (1, 20)}
PRICE_PER_M2_RANGE = (3, 2500)  # million VND / m2

PREFIXES = {
    "street": ["đường", "duong", "phố", "pho"],
    "ward": ["phường", "phuong", "xã", "xa", "thị trấn", "thi tran", "đặc khu", "dac khu"],
    "province": ["thành phố", "thanh pho", "tỉnh", "tinh", "tp"],
}
PROVINCE_ALIAS = {"hcm": "ho chi minh", "tphcm": "ho chi minh",
                  "sai gon": "ho chi minh", "hn": "ha noi"}

MIN_COUNT = 10  # street or ward that is under 10 rows in dataset -> "ít dữ liệu"
CONF_LEVELS = (0.5, 0.8, 0.9)  # prediction interval confidence levels
PRICE_BAND_EDGES = [10]  # billion VND: Separate predictions for houses under and above 10 billion VND
# Decision trees output step functions, so values higher than most training samples get capped at a SINGLE prediction
# So the model will extrapolate the tail distribution using local slope. The factor controls how much of that slope to apply
# Tuned via backtest: anchor in dense data region and evaluate on high-value properties (0 = constant prediction)
TAIL_QUANTILE = 0.995
TAIL_DAMPING = {"floors": 0.75,     # backtest: error reduction, actual prices still increase with floors > 12
                "bedrooms": 0.5,    # backtest: small impact
                "bathrooms": 0.0}   # backtest: adding bathrooms at this level doesn't increase price
LABELS = {"area": "Area", "floors": "Floors", "bedrooms": "Bedrooms",
          "bathrooms": "Bathrooms", "street": "Street", "ward": "Ward",
          "province": "Province"}


@lru_cache(maxsize=None)  # cache ward/province names for better time complexity
def normalize_place(text, kind):
    if not isinstance(text, str):
        return UNKNOWN
    t = unicodedata.normalize("NFC", text).lower() # lowercase and unaccent address
    t = re.sub(r"[^\w\s]|_", " ", t)  # TP.HCM -> tp hcm
    t = re.sub(r"\s+", " ", t).strip()

    while True:
        prefix = next((p for p in PREFIXES[kind] if t.startswith(p + " ")), None)
        if prefix is None:
            break
        t = t[len(prefix) + 1:]
        if kind != "street":  # prefix duplicates like "Đường Đường Phùng Há" because of input errors
            break

    t = unicodedata.normalize("NFD", t.replace("đ", "d"))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    if kind == "province":
        t = PROVINCE_ALIAS.get(t, t)
    return t or UNKNOWN


def address_parts(address):
    if not isinstance(address, str):
        return []
    return [p.strip() for p in address.split(",") if p.strip()]


def split_address(address):
    parts = address_parts(address)
    street = parts[0] if len(parts) >= 3 else None
    ward = parts[1] if len(parts) >= 3 else parts[0] if len(parts) == 2 else None
    province = parts[-1] if parts else None
    return (normalize_place(street, "street"), normalize_place(ward, "ward"),
            normalize_place(province, "province"))


def add_location_features(df):
    out = df.copy()
    split = pd.DataFrame([split_address(a) for a in out["address"]],
                         columns=CAT_COLS, index=out.index)
    for col in CAT_COLS:
        out[col] = split[col]
    return out


def to_model_frame(df):
    X = df.reindex(columns=FEATURES).copy()
    for col in NUM_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in CAT_COLS:
        X[col] = X[col].fillna(UNKNOWN).astype(object)
    return X


def load_dataset(csv_path):
    # read CSV, normalize addresses, drop invalid rows then return cleaned DataFrame
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in ["address", *NUM_COLS, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}. Available columns: {list(df.columns)}")

    n_start = len(df)
    for col in [*NUM_COLS, TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["address", *NUM_COLS, TARGET])
    df = add_location_features(df)

    in_range = pd.Series(True, index=df.index)
    for col, (lo, hi) in VALID_RANGE.items():
        in_range &= df[col].between(lo, hi)
    ppm = df[TARGET] * 1000 / df["area"]  # price per m2 in million VND
    bad = {
        "Invalid price (price <= 0)": df[TARGET] <= 0,
        "Out of range area/floor/room": ~in_range,
        "Abnormal price/m²": ~ppm.between(*PRICE_PER_M2_RANGE),
    }
    print(f"[data] {n_start:,} initial rows. Violations per rule (may contain overlaps):")
    for name, mask in bad.items():
        print(f"  - {name}: {int(mask.sum()):,}")
    df = df[(df[TARGET] > 0) & in_range & ppm.between(*PRICE_PER_M2_RANGE)]

    # define duplicates: same address, ward, area, price then keep only ONE row
    # because duplicates can be split in both training and testing set -> avoid data leakage
    n_valid = len(df)
    df = df.drop_duplicates(subset=["street", "ward", "province", "area", "floors", "bedrooms", "bathrooms", TARGET])
    print(f"[data] {n_valid:,} valid rows remaining -> {len(df):,} after removing duplicates")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Out-of-Distribution check
# ---------------------------------------------------------------------------
def build_reference(X_train):
    ranges = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                  "p01": float(X_train[c].quantile(0.01)), "p99": float(X_train[c].quantile(0.99))}
              for c in NUM_COLS}
    counts = {c: X_train[c].value_counts().to_dict() for c in CAT_COLS}
    counts["street_ward"] = (X_train["street"] + "|" + X_train["ward"]).value_counts().to_dict()
    return {"ranges": ranges, "counts": counts}


def check_support(row, ref):
    # check if the prediction belongs to "ok", "edge" or "out"
    out, edge = [], []
    for col in NUM_COLS:
        r, v = ref["ranges"][col], row[col]
        if v < r["min"] or v > r["max"]:
            out.append(f"{LABELS[col]} = {v:g} is OUTSIDE the training data range "
                       f"({r['min']:g} - {r['max']:g}).")
        elif v < r["p01"] or v > r["p99"]:
            edge.append(f"{LABELS[col]} = {v:g} falls in limited data region "
                        f"(99% of data is within {r['p01']:g} - {r['p99']:g}).")

    for col in CAT_COLS:
        value = row[col]
        n = ref["counts"][col].get(value, 0)
        if value == UNKNOWN:
            edge.append(f"Couldn't read {LABELS[col].lower()} from address.")
        elif n == 0:
            out.append(f"{LABELS[col]} '{value}' has never appeared in training data.")
        elif n < MIN_COUNT:
            edge.append(f"{LABELS[col]} '{value}' only has {n} listings in training data.")

    s_n = ref["counts"]["street"].get(row["street"], 0)
    w_n = ref["counts"]["ward"].get(row["ward"], 0)
    pair = f"{row['street']}|{row['ward']}"
    if s_n and w_n and ref["counts"]["street_ward"].get(pair, 0) == 0:
        edge.append(f"Street '{row['street']}' exists in data, but no listings found in"
                    f"ward '{row['ward']}'.")

    level = "out" if out else "edge" if edge else "ok"
    return level, out + edge

class PricePerM2Regressor(RegressorMixin, BaseEstimator):
    """
    Trains on log(price/m²) and then scales by area, so prediction still returns PRICE (billion VND).
    Note: This class must reside in features.py and imported by BOTH train.py and predict.py;
    otherwise, joblib cannot unpickle the saved model.
    """

    def __init__(self, regressor, tail_damping=None):
        self.regressor = regressor
        self.tail_damping = tail_damping

    def fit(self, X, y):
        ppm = np.asarray(y, dtype=float) * 1000 / X["area"].to_numpy(dtype=float)  # million per m2
        self.regressor_ = clone(self.regressor).fit(X, np.log(ppm))
        self.tail_damping_ = dict(TAIL_DAMPING if self.tail_damping is None else self.tail_damping) # extrapolation 
        self.tail_limit_ = {c: float(X[c].quantile(TAIL_QUANTILE)) for c in self.tail_damping_}
        self.tail_lo_ = {c: float(X[c].median()) for c in self.tail_damping_}
        return self

    def _ppm(self, X):
        return np.exp(self.regressor_.predict(X))

    def predict(self, X):
        X_clip = X.copy()
        for col, limit in self.tail_limit_.items():
            X_clip[col] = np.minimum(X[col].to_numpy(dtype=float), limit)
        ppm = self._ppm(X_clip)

        for col, limit in self.tail_limit_.items():
            damping = self.tail_damping_[col]
            over = X[col].to_numpy(dtype=float) > limit
            if damping == 0 or not over.any():
                continue
            at_limit = X_clip[over]
            at_low = at_limit.copy()
            at_low[col] = self.tail_lo_[col]
            slope = np.maximum(self._ppm(at_limit) - self._ppm(at_low), 0) / (limit - self.tail_lo_[col])
            ppm[over] += damping * slope * (X[col].to_numpy(dtype=float)[over] - limit)
        return ppm * X["area"].to_numpy(dtype=float) / 1000

# ---------------------------------------------------------------------------
# prediction interval (cross-conformal): measures model error on UNSEEN houses,
# then uses the error percentiles to determine the interval width.
# ---------------------------------------------------------------------------
def conformal_quantile(residuals, level):
    # small-sample adjusted quantile (n+1) for conformal prediction standard
    n = len(residuals)
    return float(np.quantile(residuals, min(np.ceil((n + 1) * level) / n, 1.0)))


def calibrate_intervals(y_true, y_oof):
    """
    y_oof = out-of-fold predictions for the training set. Errors are computed on log(price), 
    so intervals are multiplicative: [price * e^-q, price * e^q]. Each price band has its own q 
    because high-end properties have larger errors: using a single q would under-coverage expensive homes
    """
    residual = np.abs(np.log(np.asarray(y_true, dtype=float)) - np.log(y_oof))
    band = np.searchsorted(PRICE_BAND_EDGES, y_oof, side="right")
    levels = {}
    for level in CONF_LEVELS:
        qs = []
        for b in range(len(PRICE_BAND_EDGES) + 1):
            r = residual[band == b]
            qs.append(conformal_quantile(r if len(r) >= 100 else residual, level))
        levels[str(level)] = qs
    return {"band_edges": PRICE_BAND_EDGES, "levels": levels}


def predict_interval(price, calib, level):
    # Low/high bounds for a single price or an array of prices
    price = np.asarray(price, dtype=float)
    band = np.searchsorted(calib["band_edges"], price, side="right")
    q = np.asarray(calib["levels"][str(level)])[band]
    return price * np.exp(-q), price * np.exp(q)

