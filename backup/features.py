"""
Phần dùng chung cho train.py và predict.py:
đọc + làm sạch dữ liệu, chuẩn hóa địa chỉ, kiểm tra thông tin nhập vào
có nằm trong vùng dữ liệu mô hình đã học hay không.
"""

import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

TARGET = "price"  # giá, đơn vị tỷ VND
NUM_COLS = ["area", "floors", "bedrooms", "bathrooms"]
CAT_COLS = ["street", "ward", "province"]  # 3 cột này tách ra từ address
FEATURES = NUM_COLS + CAT_COLS
UNKNOWN = "unknown"

# Giới hạn "nhà thật" do mình tự đặt. Ngoài khoảng này coi là nhập sai và bỏ.
VALID_RANGE = {"area": (5, 2000), "floors": (1, 12), "bedrooms": (1, 20), "bathrooms": (1, 20)}
PRICE_PER_M2_RANGE = (5, 3000)  # triệu đồng / m²

# Tiền tố bỏ đi để 'Đường Lê Lợi', 'le loi' là một. Mỗi phần có bộ riêng:
# 'Tỉnh lộ 10' là tên ĐƯỜNG nên 'tỉnh' chỉ được bỏ ở phần tỉnh/thành.
PREFIXES = {
    "street": ["đường", "duong", "phố", "pho"],
    "ward": ["phường", "phuong", "xã", "xa", "thị trấn", "thi tran", "đặc khu", "dac khu"],
    "province": ["thành phố", "thanh pho", "tỉnh", "tinh", "tp"],
}
PROVINCE_ALIAS = {"hcm": "ho chi minh", "tphcm": "ho chi minh",
                  "sai gon": "ho chi minh", "hn": "ha noi"}

MIN_COUNT = 10  # đường/phường có ít hơn 10 tin trong dữ liệu train -> "ít dữ liệu"
CONF_LEVELS = (0.5, 0.8, 0.9)  # các mức tin cậy của khoảng dự đoán
PRICE_BAND_EDGES = [10]  # tỷ VND: hiệu chuẩn riêng cho nhà dự đoán <10 tỷ và >=10 tỷ
# Phần "đuôi" của các biến số: cây quyết định chỉ trả về các bậc thang nên khi gặp giá trị lớn
# hơn mọi giá trị đã thấy nhiều lần, nó gom tất cả vào MỘT mức giá. Từ ngưỡng này trở lên
# (TAIL_QUANTILE của dữ liệu train), giá được nối tiếp theo độ dốc cục bộ của chính mô hình.
# Hệ số 0..1 cho biết nối tiếp bao nhiêu phần của độ dốc đó. Chọn bằng backtest: neo ở vùng
# nhiều dữ liệu rồi kiểm trên các căn thật nằm phía trên (0 = giữ nguyên, đi ngang).
TAIL_QUANTILE = 0.995
TAIL_DAMPING = {"floors": 0.75,     # backtest: nối tiếp giảm sai số, giá thật vẫn tăng theo tầng
                "bedrooms": 0.5,    # backtest: ảnh hưởng nhỏ
                "bathrooms": 0.0}   # backtest: thêm phòng tắm ở mức này KHÔNG làm giá tăng
LABELS = {"area": "Diện tích", "floors": "Số tầng", "bedrooms": "Số phòng ngủ",
          "bathrooms": "Số phòng tắm", "street": "Đường", "ward": "Phường/xã",
          "province": "Tỉnh/thành"}


@lru_cache(maxsize=None)  # tên phường/tỉnh lặp lại rất nhiều, nhớ kết quả cho nhanh
def normalize_place(text, kind):
    """'Đường Đường Phạm Văn Đồng' -> 'pham van dong'. Dùng cho cả dữ liệu lẫn người dùng nhập."""
    if not isinstance(text, str):
        return UNKNOWN
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[^\w\s]|_", " ", t)  # bỏ dấu câu: 'TP.HCM' -> 'tp hcm'
    t = re.sub(r"\s+", " ", t).strip()

    # Bỏ tiền tố TRƯỚC khi bỏ dấu, vì 'phường' (tiền tố) khác 'phương' (tên riêng)
    while True:
        prefix = next((p for p in PREFIXES[kind] if t.startswith(p + " ")), None)
        if prefix is None:
            break
        t = t[len(prefix) + 1:]
        if kind != "street":  # chỉ tên đường mới bị lặp kiểu 'Đường Đường ...'
            break

    # Bỏ dấu để 'Hoà' và 'Hòa' (và người dùng gõ không dấu) khớp nhau
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
    """'đường X, phường Y, tỉnh Z' -> (street, ward, province) đã chuẩn hóa."""
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
    """Đúng các cột mô hình cần, đúng thứ tự, đúng kiểu (số -> float, chữ -> object)."""
    X = df.reindex(columns=FEATURES).copy()
    for col in NUM_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in CAT_COLS:
        X[col] = X[col].fillna(UNKNOWN).astype(object)
    return X


def load_dataset(csv_path):
    """Đọc CSV, chuẩn hóa địa chỉ, bỏ dòng không dùng được. Trả về bảng sạch."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")  # file có BOM nên dùng utf-8-sig
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in ["address", *NUM_COLS, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu cột {missing}. Các cột đang có: {list(df.columns)}")

    n_start = len(df)
    for col in [*NUM_COLS, TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["address", *NUM_COLS, TARGET])
    df = add_location_features(df)

    in_range = pd.Series(True, index=df.index)
    for col, (lo, hi) in VALID_RANGE.items():
        in_range &= df[col].between(lo, hi)
    ppm = df[TARGET] * 1000 / df["area"]  # giá mỗi m², triệu đồng
    bad = {
        "giá <= 0 (không có giá thật)": df[TARGET] <= 0,
        "diện tích/tầng/phòng ngoài giới hạn": ~in_range,
        "giá/m² bất thường": ~ppm.between(*PRICE_PER_M2_RANGE),
    }
    print(f"[data] {n_start:,} dòng ban đầu. Số dòng vi phạm từng quy tắc (có thể chồng nhau):")
    for name, mask in bad.items():
        print(f"  - {name}: {int(mask.sum()):,}")
    df = df[(df[TARGET] > 0) & in_range & ppm.between(*PRICE_PER_M2_RANGE)]

    # Tin đăng lại: cùng đường, phường, diện tích, giá. Để lại thì cùng một căn
    # có thể rơi vào cả train lẫn test và làm điểm đẹp hơn thực tế.
    n_valid = len(df)
    df = df.drop_duplicates(subset=["street", "ward", "province", "area", "floors", "bedrooms", "bathrooms", TARGET])
    print(f"[data] Còn {n_valid:,} dòng hợp lệ -> {len(df):,} sau khi bỏ tin trùng")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Kiểm tra thông tin nhập vào có nằm trong vùng dữ liệu huấn luyện không
# ---------------------------------------------------------------------------
def build_reference(X_train):
    """Bảng 'dữ liệu train có gì' (khoảng giá trị, số tin theo từng đường/phường)."""
    ranges = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                  "p01": float(X_train[c].quantile(0.01)), "p99": float(X_train[c].quantile(0.99))}
              for c in NUM_COLS}
    counts = {c: X_train[c].value_counts().to_dict() for c in CAT_COLS}
    counts["street_ward"] = (X_train["street"] + "|" + X_train["ward"]).value_counts().to_dict()
    return {"ranges": ranges, "counts": counts}


def check_support(row, ref):
    """
    Trả về (mức, danh sách lý do):
      'ok'   = nằm trong vùng dữ liệu đã học
      'edge' = trong khoảng đã học nhưng rất ít dữ liệu
      'out'  = nằm NGOÀI tập huấn luyện (mô hình phải ngoại suy)
    """
    out, edge = [], []
    for col in NUM_COLS:
        r, v = ref["ranges"][col], row[col]
        if v < r["min"] or v > r["max"]:
            out.append(f"{LABELS[col]} = {v:g} nằm NGOÀI khoảng trong dữ liệu huấn luyện "
                       f"({r['min']:g} - {r['max']:g}).")
        elif v < r["p01"] or v > r["p99"]:
            edge.append(f"{LABELS[col]} = {v:g} thuộc vùng rất ít nhà "
                        f"(99% dữ liệu nằm trong {r['p01']:g} - {r['p99']:g}).")

    for col in CAT_COLS:
        value = row[col]
        n = ref["counts"][col].get(value, 0)
        if value == UNKNOWN:
            edge.append(f"Không đọc được {LABELS[col].lower()} từ địa chỉ.")
        elif n == 0:
            out.append(f"{LABELS[col]} '{value}' chưa từng xuất hiện trong dữ liệu huấn luyện.")
        elif n < MIN_COUNT:
            edge.append(f"{LABELS[col]} '{value}' chỉ có {n} tin trong dữ liệu huấn luyện.")

    s_n = ref["counts"]["street"].get(row["street"], 0)
    w_n = ref["counts"]["ward"].get(row["ward"], 0)
    pair = f"{row['street']}|{row['ward']}"
    if s_n and w_n and ref["counts"]["street_ward"].get(pair, 0) == 0:
        edge.append(f"Đường '{row['street']}' có trong dữ liệu nhưng chưa có tin nào "
                    f"thuộc phường/xã '{row['ward']}'.")

    level = "out" if out else "edge" if edge else "ok"
    return level, out + edge

class PricePerM2Regressor(RegressorMixin, BaseEstimator):
    """
    Học log(giá/m²) rồi nhân lại diện tích, nên predict() vẫn trả về GIÁ (tỷ VND).
    Có thêm bước nối tiếp phần đuôi của số tầng / phòng ngủ / phòng tắm (xem TAIL_DAMPING).
    Class này phải nằm trong file mà CẢ train.py lẫn predict.py cùng import, nếu không
    joblib sẽ không nạp lại được mô hình đã lưu.
    """

    def __init__(self, regressor, tail_damping=None):
        self.regressor = regressor
        self.tail_damping = tail_damping

    def fit(self, X, y):
        ppm = np.asarray(y, dtype=float) * 1000 / X["area"].to_numpy(dtype=float)  # triệu/m²
        self.regressor_ = clone(self.regressor).fit(X, np.log(ppm))
        self.tail_damping_ = dict(TAIL_DAMPING if self.tail_damping is None else self.tail_damping)
        self.tail_limit_ = {c: float(X[c].quantile(TAIL_QUANTILE)) for c in self.tail_damping_}
        self.tail_lo_ = {c: float(X[c].median()) for c in self.tail_damping_}
        return self

    def _ppm(self, X):
        return np.exp(self.regressor_.predict(X))

    def predict(self, X):
        # 1) kẹp các biến đuôi về ngưỡng, nơi mô hình còn đủ dữ liệu để phân biệt
        X_clip = X.copy()
        for col, limit in self.tail_limit_.items():
            X_clip[col] = np.minimum(X[col].to_numpy(dtype=float), limit)
        ppm = self._ppm(X_clip)

        # 2) phần vượt ngưỡng: cộng thêm độ dốc cục bộ (giữa trung vị và ngưỡng) x phần vượt
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
# Khoảng dự đoán (cross-conformal): đo sai số của mô hình trên nhà nó CHƯA thấy,
# rồi dùng phân vị của sai số đó làm độ rộng khoảng.
# ---------------------------------------------------------------------------
def conformal_quantile(residuals, level):
    """Phân vị có hiệu chỉnh cỡ mẫu nhỏ (n+1) của chuẩn conformal."""
    n = len(residuals)
    return float(np.quantile(residuals, min(np.ceil((n + 1) * level) / n, 1.0)))


def calibrate_intervals(y_true, y_oof):
    """
    y_oof = giá dự đoán "ngoài fold" cho tập train. Sai số tính trên log(giá), nên
    khoảng là phép NHÂN: [giá * e^-q, giá * e^q]. Mỗi nhóm giá dự đoán có q riêng,
    vì nhà đắt sai nhiều hơn: dùng một q chung thì nhà đắt bị phủ thiếu.
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
    """Khoảng (thấp, cao) cho một giá hoặc cả mảng giá."""
    price = np.asarray(price, dtype=float)
    band = np.searchsorted(calib["band_edges"], price, side="right")
    q = np.asarray(calib["levels"][str(level)])[band]
    return price * np.exp(-q), price * np.exp(q)

# ----- Stacking: 2 learners independent, Ridge on out-of-fold log prices -----
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor


class CatBoostPerM2(RegressorMixin, BaseEstimator):
    """CatBoost native categoricals; same log(price/m²) target as HistGB."""
    def __init__(self, iterations=850, depth=7, learning_rate=0.045, l2_leaf_reg=5.0,
                 thread_count=4, random_state=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.thread_count = thread_count
        self.random_state = random_state

    def fit(self, X, y):
        X = to_model_frame(X)
        target = np.log(np.asarray(y, float) * 1000 / X['area'].to_numpy())
        self.regressor_ = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
            cat_features=CAT_COLS, thread_count=self.thread_count,
            random_seed=self.random_state, loss_function='MAE',
            verbose=False, allow_writing_files=False)
        self.regressor_.fit(X, target)
        return self

    def predict(self, X):
        X = to_model_frame(X)
        return np.exp(self.regressor_.predict(X)) * X['area'].to_numpy() / 1000


class StackedHouseRegressor(RegressorMixin, BaseEstimator):
    """Leakage-safe OOF ridge stack; all models refit on whole fitting split."""
    def __init__(self, hist_model, cat_model, n_folds=4, ridge_alpha=2.0,
                 random_state=42):
        self.hist_model = hist_model
        self.cat_model = cat_model
        self.n_folds = n_folds
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    def fit(self, X, y):
        X = to_model_frame(X)
        y = np.asarray(y, float)
        cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        # Critical: Ridge sees only predictions from models that never saw this row.
        hist_oof = cross_val_predict(clone(self.hist_model), X, y, cv=cv, n_jobs=1)
        cat_oof = cross_val_predict(clone(self.cat_model), X, y, cv=cv, n_jobs=1)
        log_base = np.column_stack([np.log(np.maximum(hist_oof, 1e-8)),
                                    np.log(np.maximum(cat_oof, 1e-8))])
        self.meta_ = Ridge(alpha=self.ridge_alpha, positive=True)
        self.meta_.fit(log_base, np.log(y))
        self.hist_ = clone(self.hist_model).fit(X, y)
        self.cat_ = clone(self.cat_model).fit(X, y)
        return self

    def predict_components(self, X):
        X = to_model_frame(X)
        return (np.maximum(self.hist_.predict(X), 1e-8),
                np.maximum(self.cat_.predict(X), 1e-8))

    def predict(self, X):
        p1, p2 = self.predict_components(X)
        log_input = np.column_stack([np.log(p1), np.log(p2)])
        return np.exp(self.meta_.predict(log_input))


class TinyAreaCorrection(RegressorMixin, BaseEstimator):
    """Optional, externally validated correction. Disabled by default.

    If validated, use gentle log-price uplift fading to zero at 15 m².
    Never activate based on a few anecdotes or on the test split.
    """
    def __init__(self, base_model, log_uplift=0.0, threshold=15.0):
        self.base_model = base_model
        self.log_uplift = log_uplift
        self.threshold = threshold

    def fit(self, X, y):
        self.base_ = clone(self.base_model).fit(X, y)
        return self

    def predict(self, X):
        return self.apply(self.base_.predict(X), X)

    def apply(self, predictions, X):
        area = np.asarray(X['area'], float)
        strength = np.clip((self.threshold - area) / 5.0, 0, 1)
        return np.asarray(predictions) * np.exp(self.log_uplift * strength)
