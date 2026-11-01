"""
Predicts property prices using a pretrained model (only predict)
Requires running 'python -m src.train' at least once beforehand.

    python -m src.predict        # run sample examples in this file
    python -m src.predict --address "Đường D1, phường Trung An, Đồng Tháp" --area 60 --floors 3 --bedrooms 3 --bathrooms 3
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.features import (LABELS, NUM_COLS, add_location_features, address_parts,
                          check_support, predict_interval, to_model_frame)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
META_PATH = ROOT / "models" / "meta.json"
REF_PATH = ROOT / "models" / "reference.json"

SHOW_LEVELS = (0.8, 0.5)  #intervals to print (wider intervals first)

LEVEL_TEXT = {"ok": "in distribution",
              "edge": "limited data",
              "out": "OUT of distribution"}


def load_artifacts():
    if not all(p.exists() for p in (MODEL_PATH, META_PATH, REF_PATH)):
        sys.exit("Model not found. Please run first: python -m src.train")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if "conformal" not in meta:
        sys.exit("Model trained with an older version (prediction intervals missing). "
                 "Please re-run: python -m src.train")
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    return model, meta, ref


def predict_price(model, ref, meta, address, area, floors, bedrooms, bathrooms):
    """Predicts price (billion VND). Always output a value, along with data support level and reasoning."""
    row = pd.DataFrame([{"address": address, "area": area, "floors": floors,
                         "bedrooms": bedrooms, "bathrooms": bathrooms}])
    X = to_model_frame(add_location_features(row))
    price = float(model.predict(X)[0])

    record = X.iloc[0].to_dict()
    level, reasons = check_support(record, ref)
    if len(address_parts(address)) != 3:
        reasons.insert(0, "Address format should be 'street, ward, city/province'; "
                          "current address reading might be incorrect.")
        level = "edge" if level == "ok" else level
    # keep the original HistGB tail extrapolation warning
    for col, limit in model.tail_limit_.items():
        if record[col] > limit:
            if model.tail_damping_[col] > 0:
                how = 'extrapolated using local slope (unverified)'
            else:
                how = 'capped at the learned threshold'
            reasons.append(f"{LABELS[col]} = {record[col]:g} exceeds threshold {limit:g}: {how}.")
    # out-of-bounds data (e.g. area 2500 m2)
    beyond = [LABELS[c].lower() for c in NUM_COLS
              if record[c] < ref["ranges"][c]["min"] or record[c] > ref["ranges"][c]["max"]]
    intervals = {lv: tuple(float(v) for v in predict_interval(price, meta["conformal"], lv))
                 for lv in SHOW_LEVELS}
    return {"price": price, "level": level, "reasons": reasons, "beyond": beyond,
            "intervals": intervals,
            "parsed": (record["street"], record["ward"], record["province"])}


def print_result(result, meta):
    price, level = result["price"], result["level"]
    street, ward, province = result["parsed"]
    print(f"  Parsed address : street={street} | ward={ward} | province={province}")
    print(f"  Predicted price: {price:.2f} {meta['price_unit']} (~{price * 1000:,.0f} million VND)")
    conf = meta["conformal"]
    for lv in SHOW_LEVELS:
        lo, hi = result["intervals"][lv]
        if result["beyond"]:
            note = "no longer guaranteed as data significantly exceeds training range"
        else:
            # empirical coverage measured on the test set: at what percentage of actual prices truly fall within this interval
            cov = conf["test_coverage80_by_support_pct"].get(level) if lv == 0.8 else None
            cov = cov if cov is not None else conf["test_coverage_pct"][str(lv)]
            note = f"test set covers actual price for {cov:.0f}% of similar properties"
        print(f"  {lv:.0%} interval     : {lo:.2f} - {hi:.2f} {meta['price_unit']} ({note})")
    print(f"  Data scope : {LEVEL_TEXT[level]}")

    ref_err = meta["error_by_support"].get(level)
    if result["beyond"]:
        # the 'out-of-training-data' group in the test set contains only unseen streets/wards
        # so it does not represent this case -> exclude reference errors to avoid misleading results
        print(f"  Reference error: NOT applicable ({', '.join(result['beyond'])} fall outside "
              "training data range; prices there are extrapolated estimates, unverified).")
    elif ref_err:
        note = " (small sample, for reference only)" if ref_err["n"] < 100 else ""
        print(f"  Reference error (this group, test set, n={ref_err['n']:,}){note}: "
              f"MedAPE: {ref_err['MedAPE_pct']:.0f}%, "
              f"{ref_err['Within20_pct']:.0f}% of properties deviate by no more than 20%")
    for r in result["reasons"]:
        print(f"  Note: {r}")
    if level == "out":
        print("  Disclaimer: contains information outside the training set, the model must extrapolate beyond "
              "what it has learned. Should be considered as reference only; error may exceed the reference level.")
    elif level == "edge":
        print("  Disclaimer: data in this region is sparse, so error may be higher than the general level.")


EXAMPLES = [
    dict(address="Đường Lê Văn Sỹ, Phường Phú Nhuận, Thành phố Hồ Chí Minh",
         area=60, floors=3, bedrooms=3, bathrooms=3),
    dict(address="đường Hoàng Quốc Việt, phường Nghĩa Đô, TP. Hà Nội",
         area=45, floors=5, bedrooms=4, bathrooms=4),
    dict(address="Đường Abc Xyz, Xã Không Có, Tỉnh Cà Mau",
         area=2501, floors=15, bedrooms=3, bathrooms=2),
]


def main():
    model, meta, ref = load_artifacts()
    print(f"Model: {meta['model_name']} (trained at {meta['trained_at']}). "
          f"Test: MedAPE {meta['test_metrics']['MedAPE_pct']:.0f}%, "
          f"{meta['test_metrics']['Within20_pct']:.0f}% of properties within 20%")

    if len(sys.argv) == 1:  # run the built-in example if no arguments are provided
        for i, ex in enumerate(EXAMPLES, 1):
            print(f"\nExample {i}: {ex['address']} | {ex['area']} m²")
            print_result(predict_price(model, ref, meta, **ex), meta)
        return

    p = argparse.ArgumentParser(description="Input_house_price_prediction")
    p.add_argument("--address", required=True, help="\"Street X, Ward Y, Province Z\"")
    for name, label in [("area", "House area (in m²)"), ("floors", "Number of floors"),
                        ("bedrooms", "Number of bedrooms"), ("bathrooms", "Number of bathrooms")]:
        p.add_argument(f"--{name}", required=True, type=float, help=label)
    args = vars(p.parse_args())

    print(f"\n{args['address']} | {args['area']:g} m²")
    print_result(predict_price(model, ref, meta, **args), meta)


if __name__ == "__main__":
    main()