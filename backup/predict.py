"""
Dự đoán giá một căn nhà bằng mô hình đã train (không học gì thêm).
Cần chạy  python -m src.train  ít nhất một lần trước.

  python -m src.predict        # chạy các ví dụ mẫu
  python -m src.predict --address "Đường Lê Văn Sỹ, Phường Phú Nhuận, Thành phố Hồ Chí Minh" --area 60 --floors 3 --bedrooms 3 --bathrooms 3
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

SHOW_LEVELS = (0.8, 0.5)  # các khoảng sẽ in ra (khoảng rộng trước)

LEVEL_TEXT = {"ok": "trong vùng dữ liệu đã học",
              "edge": "vùng ít dữ liệu",
              "out": "NGOÀI tập huấn luyện"}


def load_artifacts():
    if not all(p.exists() for p in (MODEL_PATH, META_PATH, REF_PATH)):
        sys.exit("Chưa có mô hình. Hãy chạy trước:  python -m src.train")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if "conformal" not in meta:
        sys.exit("Mô hình được train bằng bản cũ (chưa có khoảng dự đoán). "
                 "Hãy chạy lại:  python -m src.train")
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    return model, meta, ref


def predict_price(model, ref, meta, address, area, floors, bedrooms, bathrooms):
    """Dự đoán giá (tỷ VND). Luôn đưa ra con số, kèm mức hỗ trợ của dữ liệu và lý do."""
    row = pd.DataFrame([{"address": address, "area": area, "floors": floors,
                         "bedrooms": bedrooms, "bathrooms": bathrooms}])
    X = to_model_frame(add_location_features(row))  # cùng hàm với lúc train
    price = float(model.predict(X)[0])

    record = X.iloc[0].to_dict()
    level, reasons = check_support(record, ref)
    if len(address_parts(address)) != 3:
        reasons.insert(0, "Địa chỉ nên có dạng 'đường, phường/xã, tỉnh/thành'; "
                          "cách đọc địa chỉ hiện tại có thể bị sai.")
        level = "edge" if level == "ok" else level
    # HistGB alone has the original tail extrapolation. Stacking may include it
    # as one component; do not falsely describe the whole stack as using that rule.
    hist_part = model.hist_ if hasattr(model, 'hist_') else model
    if hasattr(hist_part, 'tail_limit_'):
        for col, limit in hist_part.tail_limit_.items():
            if record[col] > limit:
                reasons.append(f"{LABELS[col]} = {record[col]:g} > {limit:g}: "
                               "HistGB applies its original tail rule; "
                               "the ensemble output is not validated here.")
    if record['area'] < 15:
        reasons.append("Nhà siêu nhỏ (<15 m²): kiểm tra riêng sai số backtest; "
                       "chưa tự động cộng 'premium' vì thiếu bằng chứng độc lập đủ mạnh.")
    # Số liệu vượt khỏi khoảng đã học (vd diện tích 5000 m²)
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
    print(f"  Địa chỉ đọc được: đường={street} | phường/xã={ward} | tỉnh/thành={province}")
    print(f"  Giá dự đoán     : {price:.2f} {meta['price_unit']} (~{price * 1000:,.0f} triệu)")
    conf = meta["conformal"]
    for lv in SHOW_LEVELS:
        lo, hi = result["intervals"][lv]
        if result["beyond"]:
            note = "không còn đảm bảo vì số liệu vượt xa vùng đã học"
        else:
            # độ phủ thực tế đo trên tập test: khoảng này thật sự chứa giá thật ở bao nhiêu % số căn
            cov = conf["test_coverage80_by_support_pct"].get(level) if lv == 0.8 else None
            cov = cov if cov is not None else conf["test_coverage_pct"][str(lv)]
            note = f"trên tập test chứa giá thật ở {cov:.0f}% số căn thuộc nhóm tương tự"
        print(f"  Khoảng {lv:.0%}      : {lo:.2f} - {hi:.2f} {meta['price_unit']} ({note})")
    print(f"  Phạm vi dữ liệu : {LEVEL_TEXT[level]}")

    ref_err = meta["error_by_support"].get(level)
    if result["beyond"]:
        # Nhóm 'ngoài tập huấn luyện' trong test chỉ gồm đường/phường lạ nên không đại
        # diện cho trường hợp này -> không đưa sai số tham chiếu để tránh gây hiểu lầm.
        print(f"  Sai số tham chiếu: KHÔNG áp dụng được ({', '.join(result['beyond'])} nằm ngoài "
              "khoảng trong dữ liệu huấn luyện; giá ở đó là ước lượng ngoại suy, chưa được kiểm chứng).")
    elif ref_err:
        note = " (mẫu nhỏ, chỉ tham khảo)" if ref_err["n"] < 100 else ""
        print(f"  Sai số tham chiếu (nhóm này, tập test, n={ref_err['n']:,}){note}: "
              f"lệch trung vị {ref_err['MedAPE_pct']:.0f}%, "
              f"{ref_err['Within20_pct']:.0f}% số căn lệch không quá 20%")
    for r in result["reasons"]:
        print(f"  Lưu ý: {r}")
    if level == "out":
        print("  Disclaimer: có thông tin nằm ngoài tập huấn luyện, mô hình phải suy đoán ngoài "
              "những gì đã học. Chỉ nên xem là con số tham khảo; sai số có thể lớn hơn mức tham chiếu.")
    elif level == "edge":
        print("  Disclaimer: dữ liệu ở vùng này còn mỏng nên sai số có thể lớn hơn mức chung.")


EXAMPLES = [
    dict(address="Đường Lê Văn Sỹ, Phường Phú Nhuận, Thành phố Hồ Chí Minh",
         area=60, floors=3, bedrooms=3, bathrooms=3),
    dict(address="đường Hoàng Quốc Việt, phường Nghĩa Đô, TP. Hà Nội",
         area=45, floors=5, bedrooms=4, bathrooms=4),
    dict(address="Đường Abc Xyz, Xã Không Có, Tỉnh Cà Mau",
         area=2500, floors=15, bedrooms=3, bathrooms=2),
]


def main():
    model, meta, ref = load_artifacts()
    print(f"Mô hình: {meta['model_name']} (train lúc {meta['trained_at']}). "
          f"Test: lệch trung vị {meta['test_metrics']['MedAPE_pct']:.0f}%, "
          f"{meta['test_metrics']['Within20_pct']:.0f}% số căn lệch không quá 20%")

    if len(sys.argv) == 1:  # không truyền tham số -> chạy ví dụ mẫu
        for i, ex in enumerate(EXAMPLES, 1):
            print(f"\nVí dụ {i}: {ex['address']} | {ex['area']} m²")
            print_result(predict_price(model, ref, meta, **ex), meta)
        return

    p = argparse.ArgumentParser(description="Dự đoán giá nhà")
    p.add_argument("--address", required=True, help="'đường X, phường/xã Y, tỉnh/thành Z'")
    for name, label in [("area", "Diện tích (m²)"), ("floors", "Số tầng"),
                        ("bedrooms", "Số phòng ngủ"), ("bathrooms", "Số phòng tắm")]:
        p.add_argument(f"--{name}", required=True, type=float, help=label)
    args = vars(p.parse_args())

    print(f"\n{args['address']} | {args['area']:g} m²")
    print_result(predict_price(model, ref, meta, **args), meta)


if __name__ == "__main__":
    main()