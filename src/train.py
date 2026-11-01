#Train from project root folder: python -m src.train (Python 3.10+)

import json
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import (KFold, RandomizedSearchCV, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.features import (CAT_COLS, CONF_LEVELS, FEATURES, NUM_COLS, TARGET,
                          PricePerM2Regressor,
                          build_reference, calibrate_intervals, check_support,
                          load_dataset, predict_interval, to_model_frame)

# HYPERPARAMETERS & CONFIGURATION
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, MODEL_DIR, REPORT_DIR = ROOT / 'dataset', ROOT / 'models', ROOT / 'reports'
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
RUN_TUNING = True      #optional: tune HistGB using train only (expensive)
TUNE_ITERATIONS = 16
N_JOBS = -2            #limit RAM usage, -1 for best performance


def medape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.median(np.abs(p - y) / y) * 100)


def within20(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(p - y) / y <= .20) * 100)


def metrics(y, p):
    return {'MedAPE_pct': medape(y, p), 'Within20_pct': within20(y, p)}


SCORING = {'medape': make_scorer(medape, greater_is_better=False),
           'within20': make_scorer(within20)}


def find_csv():
    main = DATA_DIR / 'vietnam_real_estate_refined_v4.csv'
    if main.exists():
        return main
    found = sorted(DATA_DIR.glob('*.csv'))
    if not found:
        raise FileNotFoundError('Not found vietnam_real_estate_refined_v4.csv in dataset/')
    return found[0]


def save_fig(filename):
    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=130)
    plt.close()


def hist_model():
    prep = ColumnTransformer([
        ('num', 'passthrough', NUM_COLS),
        ('loc', TargetEncoder(target_type='continuous', random_state=SEED), CAT_COLS)])
    hist = HistGradientBoostingRegressor(
        loss='absolute_error', learning_rate=.05, max_iter=1200,
        max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.,
        random_state=SEED)
    return PricePerM2Regressor(Pipeline([('prep', prep), ('reg', hist)]))


def save_eda(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(np.log10(df[TARGET]), bins=60)
    ax[0].set(title='log10(price, billion VND)')
    ax[1].hist(np.log10(df['area']), bins=60)
    ax[1].set(title='log10(area, m²)')
    ppm = (df[TARGET] * 1000 / df['area']).groupby(df['ward']).agg(['median', 'size'])
    top = ppm[ppm['size'] >= 200].sort_values('median').tail(15)
    if len(top):
        top['median'].plot.barh(ax=ax[2], title='Median million VND/m² by Ward')
    save_fig('eda.png')



def main():
    start = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    print('=' * 68)
    print('  VIETNAM REAL ESTATE | HISTGB TRAINING REPORT')
    print('=' * 68)
    print('\n[1/5] Data preparation')
    df = load_dataset(find_csv())
    print(f'  Valid listings   : {len(df):,}')
    save_eda(df)
    X, y = to_model_frame(df), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f'  Split train/test : {len(X_train):,} / {len(X_test):,} ({TEST_SIZE:.0%} holdout)')

    hist = hist_model()
    if RUN_TUNING:
        print('\n[2/5] Hyperparameter tuning (training data only)')
        space = {'regressor__reg__max_leaf_nodes': [255, 383, 511],
                 'regressor__reg__min_samples_leaf': [20, 30, 40],
                 'regressor__reg__learning_rate': [0.04, 0.05, 0.06],
                 'regressor__reg__l2_regularization': [7.0, 10.0, 13.0]}
        search = RandomizedSearchCV(hist, space, n_iter=TUNE_ITERATIONS,
                    cv=CV_FOLDS, scoring=SCORING['medape'], random_state=SEED, n_jobs=N_JOBS)
        search.fit(X_train, y_train)
        hist = search.best_estimator_
        print('  Best parameters:', search.best_params_)

    baselines = {'Baseline (median price)': DummyRegressor(strategy='median'),
                 'Baseline (median price/m²)': PricePerM2Regressor(DummyRegressor(strategy='median'))}
    ridge_prep = ColumnTransformer([('num', StandardScaler(), NUM_COLS),
                                    ('loc', OneHotEncoder(handle_unknown='ignore'), CAT_COLS)])
    ridge_baseline = PricePerM2Regressor(Pipeline([('prep', ridge_prep),
                                                    ('reg', Ridge(alpha=1.0))]))
    candidates = {**baselines, 'Ridge + one-hot': ridge_baseline,
                  'HistGB': hist}
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    print(f'\n[2/5] Cross-validation ({CV_FOLDS} folds)')
    print(f"  {'Model':<31} {'MedAPE':>9} {'Within20':>10} {'Fit (s)':>10}")
    print('  ' + '-' * 63)
    for name, est in candidates.items():
        res = cross_validate(est, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        row = {'model': name, 'MedAPE_pct': -res['test_medape'].mean(),
               'MedAPE_std': res['test_medape'].std(),
               'Within20_pct': res['test_within20'].mean(),
               'fit_time_s': res['fit_time'].mean()}
        rows.append(row)
        print(f"  {name:<31} {row['MedAPE_pct']:>8.2f}% {row['Within20_pct']:>9.2f}% {row['fit_time_s']:>10.1f}")
    cv_table = pd.DataFrame(rows).set_index('model')
    cv_table.round(3).to_csv(REPORT_DIR / 'cv_results.csv', encoding='utf-8-sig')
    # HistGB is fixed as the final model, keeping Ridge only as a benchmark.
    best_name = 'HistGB'
    print(f'\n[3/5] Final model: {best_name} (fixed selection)')
    # Evaluate HistGB on untouched holdout.
    final = clone(candidates[best_name]).fit(X_train, y_train)
    pred_train, pred_test = final.predict(X_train), final.predict(X_test)
    train_m, test_m = metrics(y_train, pred_train), metrics(y_test, pred_test)
    print(f"  {'Dataset':<17} {'MedAPE':>10} {'Within20':>11}")
    print('  ' + '-' * 40)
    for label, result in [('Training', train_m), ('Holdout test', test_m)]:
        print(f"  {label:<17} {result['MedAPE_pct']:>9.2f}% {result['Within20_pct']:>10.2f}%")

    base_m = {n: metrics(y_test, clone(est).fit(X_train, y_train).predict(X_test))
              for n, est in baselines.items()}
    ref = build_reference(X_train)
    levels = np.array([check_support(r, ref)[0] for r in X_test.to_dict('records')])
    by_support = {}
    for lv in ('ok', 'edge', 'out'):
        m = levels == lv
        if m.any():
            by_support[lv] = {'n': int(m.sum()), **metrics(y_test.to_numpy()[m], pred_test[m])}
    print('\n  Holdout metrics by data support:')
    print(f"  {'Support':<12} {'Count':>8} {'MedAPE':>10} {'Within20':>11}")
    for lv, result in by_support.items():
        print(f"  {lv:<12} {result['n']:>8,} {result['MedAPE_pct']:>9.2f}% {result['Within20_pct']:>10.2f}%")

    # Conformal fitted on train OOF only; untouched test measures actual coverage.
    oof = cross_val_predict(clone(candidates[best_name]), X_train, y_train,
                           cv=cv, n_jobs=1)
    calib = calibrate_intervals(y_train, oof)
    y_arr = y_test.to_numpy()
    coverage = {}
    for level in CONF_LEVELS:
        low, high = predict_interval(pred_test, calib, level)
        coverage[str(level)] = float(np.mean((y_arr >= low) & (y_arr <= high)) * 100)
    low, high = predict_interval(pred_test, calib, .8)
    hit80 = (y_arr >= low) & (y_arr <= high)
    cov_support = {lv: float(hit80[levels == lv].mean() * 100) for lv in by_support}
    band = np.searchsorted(calib['band_edges'], pred_test, side='right')
    cov_band = {f'band{b}': float(hit80[band == b].mean() * 100) for b in np.unique(band)}
    print('\n[4/5] Prediction interval coverage (holdout)')
    for level in CONF_LEVELS:
        print(f"  Target {level:>4.0%}  ->  Observed {coverage[str(level)]:>6.2f}%")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_test, pred_test, s=3, alpha=.2)
    lim = [float(min(y_test.min(), pred_test.min())),
           float(max(y_test.max(), pred_test.max()))]
    ax[0].plot(lim, lim, 'r--')
    ax[0].set(xscale='log', yscale='log', xlabel='Actual (billion VND)',
              ylabel='Predicted (billion VND)')
    ax[1].hist(np.clip((pred_test - y_arr) / y_arr * 100, -100, 200), bins=60)
    ax[1].set(xlabel='Signed error (%)')
    save_fig('error_analysis.png')
    worst = df.loc[X_test.index, ['address', 'area', 'floors', TARGET]].copy()
    worst['predicted'] = pred_test
    worst['absolute_error'] = abs(pred_test - y_arr)
    worst.nlargest(30, 'absolute_error').to_csv(
        REPORT_DIR / 'worst_predictions.csv', index=False, encoding='utf-8-sig')
    # Keep permutation importance reports, without printing a status message.
    imp = permutation_importance(final, X_test, y_test, scoring=SCORING['medape'],
                                 n_repeats=3, random_state=SEED, n_jobs=1)
    importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    importance.to_csv(REPORT_DIR / 'feature_importance.csv',
                      header=['MedAPE_increase'], encoding='utf-8-sig')
    importance.plot.barh(figsize=(7, 4), title='MedAPE increase when feature shuffled')
    save_fig('feature_importance.png')

    joblib.dump(final, MODEL_DIR / 'house_price_model.joblib')
    (MODEL_DIR / 'reference.json').write_text(
        json.dumps(ref, ensure_ascii=False), encoding='utf-8')
    meta = {
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'model_name': best_name, 'price_unit': 'billion VND', 'features': FEATURES,
        'n_train': len(X_train), 'n_test': len(X_test),
        'train_metrics': train_m, 'test_metrics': test_m,
        'baseline_test_metrics': base_m, 'error_by_support': by_support,
        'conformal': {**calib, 'test_coverage_pct': coverage,
                      'test_coverage80_by_support_pct': cov_support,
                      'test_coverage80_by_price_band_pct': cov_band,
                      'median_width_ratio_80': float(np.median(high / low))},
        'versions': {'python': platform.python_version(),
                     'sklearn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__}}
    (MODEL_DIR / 'meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n[5/5] Saved outputs')
    print(f'  Model   : {MODEL_DIR / "house_price_model.joblib"}')
    print(f'  Reports : {REPORT_DIR}')
    print(f'  Runtime : {time.time() - start:.0f} seconds')
    print('=' * 68)


if __name__ == '__main__':
    main()
