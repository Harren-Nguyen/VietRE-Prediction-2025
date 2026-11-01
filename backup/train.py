"""Train from project root: python -m src.train (Python 3.10+)."""
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
from catboost import __version__ as catboost_version

from src.features import (CAT_COLS, CONF_LEVELS, FEATURES, NUM_COLS, TARGET,
                          CatBoostPerM2, StackedHouseRegressor, PricePerM2Regressor,
                          build_reference, calibrate_intervals, check_support,
                          load_dataset, predict_interval, to_model_frame)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, MODEL_DIR, REPORT_DIR = ROOT / 'dataset', ROOT / 'models', ROOT / 'reports'
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3           # outer CV for unbiased *training* model comparison; slow for stack
STACK_FOLDS = 4        # inner CV for Ridge meta training
RUN_TUNING = True     # optional: tune HistGB using train only (expensive)
TUNE_ITERATIONS = 8
N_JOBS = 6             # limit RAM usage / oversubscription

# External holdout must be manually verified; DO NOT auto-scrape and silently train from ads.
# CSV: address,area,floors,bedrooms,bathrooms,price,source_url,verified_date
EXTERNAL_CSV = DATA_DIR / 'tiny_external_verified.csv'
MIN_EXTERNAL = 20
MIN_TINY_INTERNAL = 20


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
    found = [p for p in sorted(DATA_DIR.glob('*.csv')) if p.name != EXTERNAL_CSV.name]
    if not found:
        raise FileNotFoundError('Place vietnam_real_estate_refined_v4.csv in dataset/')
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


def cat_model():
    return CatBoostPerM2(iterations=850, depth=7, learning_rate=.045,
                         l2_leaf_reg=5, thread_count=N_JOBS, random_state=SEED)


def make_stack(hist, cat):
    return StackedHouseRegressor(hist_model=hist, cat_model=cat,
                                 n_folds=STACK_FOLDS, ridge_alpha=2., random_state=SEED)


def save_eda(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(np.log10(df[TARGET]), bins=60)
    ax[0].set(title='log10(price, billion VND)')
    ax[1].hist(np.log10(df['area']), bins=60)
    ax[1].set(title='log10(area, m²)')
    ppm = (df[TARGET] * 1000 / df['area']).groupby(df['ward']).agg(['median', 'size'])
    top = ppm[ppm['size'] >= 200].sort_values('median').tail(15)
    if len(top):
        top['median'].plot.barh(ax=ax[2], title='Median million VND/m² by ward')
    save_fig('eda.png')


def external_backtest(model):
    """Audits internet ads already independently verified and supplied as CSV.

    External results are *not* used for training, selection on test, or leakage.
    A correction is switched on only when evidence supports it, not assumed.
    """
    report = {'enabled': False, 'reason': 'No sufficiently verified external benchmark',
              'log_uplift': 0., 'external_n': 0}
    if not EXTERNAL_CSV.exists():
        return report, pd.DataFrame()
    ads = pd.read_csv(EXTERNAL_CSV, encoding='utf-8-sig')
    required = ['address', 'area', 'floors', 'bedrooms', 'bathrooms', 'price',
                'source_url', 'verified_date']
    if any(c not in ads.columns for c in required):
        report['reason'] = f'External CSV must contain: {required}'
        return report, pd.DataFrame()
    ads = ads.dropna(subset=required).copy()
    ads = ads[ads['source_url'].astype(str).str.startswith(('https://', 'http://'))]
    # Require complete actual comparable homes; advertised prices are not transactions.
    ads = ads[(pd.to_numeric(ads['area'], errors='coerce') < 15) &
              (pd.to_numeric(ads['area'], errors='coerce') >= 5)]
    ads = ads.drop_duplicates(subset=['source_url'])
    if not len(ads):
        report['reason'] = 'No eligible external ads (5 <= area < 15)'
        return report, pd.DataFrame()
    X = to_model_frame(__import__('src.features', fromlist=['add_location_features']).add_location_features(ads))
    prediction = model.predict(X)
    result = ads.copy()
    result['predicted_billion_vnd'] = prediction
    result['log_residual'] = np.log(ads['price'].to_numpy(dtype=float) / prediction)
    result['absolute_pct_error'] = np.abs(prediction - ads['price'].to_numpy(float)) / ads['price'].to_numpy(float) * 100
    report['external_n'] = len(result)
    report['external_medape_pct'] = medape(ads['price'], prediction)
    report['external_within20_pct'] = within20(ads['price'], prediction)
    report['median_log_residual'] = float(np.median(result['log_residual']))
    if len(result) < MIN_EXTERNAL:
        report['reason'] = f'Only {len(result)} external ads, need {MIN_EXTERNAL}'
        return report, result
    residuals = result['log_residual'].to_numpy()
    # Conservative confidence check: bootstrap CI for median residual must be >0.
    rng = np.random.default_rng(SEED)
    bootstrap = np.median(rng.choice(residuals, (2000, len(residuals)), replace=True), axis=1)
    lo = float(np.quantile(bootstrap, .025))
    report['median_residual_ci_low95'] = lo
    if lo <= 0:
        report['reason'] = 'No statistically convincing positive tiny-home premium'
        return report, result
    # Do not deploy an adjustment automatically based only on advertised asking prices.
    report['reason'] = ('Positive premium suggested, but correction NOT deployed: '
                        'requires independent prospective verification')
    return report, result


def main():
    start = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = load_dataset(find_csv())
    print('[data] after cleaning', len(df), 'tiny <15:', int((df.area < 15).sum()))
    save_eda(df)
    X, y = to_model_frame(df), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    print('[split]', len(X_train), 'train /', len(X_test), 'test')

    hist, cat = hist_model(), cat_model()
    if RUN_TUNING:
        print('[tune] HistGB; done only on training rows')
        space = {'regressor__reg__max_leaf_nodes': [31, 63, 127],
                 'regressor__reg__min_samples_leaf': [20, 40, 80],
                 'regressor__reg__learning_rate': [.03, .05, .08],
                 'regressor__reg__l2_regularization': [0., 1., 5.]}
        search = RandomizedSearchCV(hist, space, n_iter=TUNE_ITERATIONS,
                    cv=3, scoring=SCORING['medape'], random_state=SEED, n_jobs=N_JOBS)
        search.fit(X_train, y_train)
        hist = search.best_estimator_
        print('[tune] best:', search.best_params_)

    baselines = {'Baseline (giá trung vị)': DummyRegressor(strategy='median'),
                 'Baseline (giá/m² trung vị)': PricePerM2Regressor(DummyRegressor(strategy='median'))}
    ridge_prep = ColumnTransformer([('num', StandardScaler(), NUM_COLS),
                                    ('loc', OneHotEncoder(handle_unknown='ignore'), CAT_COLS)])
    ridge_baseline = PricePerM2Regressor(Pipeline([('prep', ridge_prep),
                                                    ('reg', Ridge(alpha=1.0))]))
    candidates = {**baselines, 'Ridge + one-hot': ridge_baseline,
                  'HistGB': hist, 'CatBoost': cat,
                  'HistGB + CatBoost + Ridge': make_stack(hist, cat)}
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    print('[cv] Outer CV; stacking also runs inner OOF folds (this may be slow).')
    for name, est in candidates.items():
        res = cross_validate(est, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        row = {'model': name, 'MedAPE_pct': -res['test_medape'].mean(),
               'MedAPE_std': res['test_medape'].std(),
               'Within20_pct': res['test_within20'].mean(),
               'fit_time_s': res['fit_time'].mean()}
        rows.append(row)
        print(f"  {name}: MedAPE={row['MedAPE_pct']:.2f}% Within20={row['Within20_pct']:.2f}%")
    cv_table = pd.DataFrame(rows).set_index('model')
    cv_table.round(3).to_csv(REPORT_DIR / 'cv_results.csv', encoding='utf-8-sig')
    # Favor ensemble only if its CV MedAPE improves; do not assume stacking wins.
    best_name = cv_table.drop(index=list(baselines))['MedAPE_pct'].idxmin()
    print('[final] CV winner:', best_name)
    # Benchmark both bases and stack on one untouched holdout, regardless of winner.
    final = clone(candidates[best_name]).fit(X_train, y_train)
    pred_train, pred_test = final.predict(X_train), final.predict(X_test)
    train_m, test_m = metrics(y_train, pred_train), metrics(y_test, pred_test)
    print('[test] train:', train_m, '| holdout:', test_m)

    base_m = {n: metrics(y_test, clone(est).fit(X_train, y_train).predict(X_test))
              for n, est in baselines.items()}
    # Tiny-house backtest on untouched internal holdout; no use to activate an uplift.
    tiny = X_test['area'].to_numpy(float) < 15
    tiny_internal = {'n': int(tiny.sum())}
    if tiny.any():
        tiny_internal.update(metrics(y_test.to_numpy()[tiny], pred_test[tiny]))
    print('[tiny] held-out <15m²:', tiny_internal)
    external, ext_predictions = external_backtest(final)
    print('[tiny] external audit:', external['reason'])
    if len(ext_predictions):
        ext_predictions.to_csv(REPORT_DIR / 'tiny_external_backtest.csv',
                               index=False, encoding='utf-8-sig')
    (REPORT_DIR / 'tiny_backtest.json').write_text(
        json.dumps({'internal_holdout': tiny_internal, 'external': external},
                   ensure_ascii=False, indent=2), encoding='utf-8')

    ref = build_reference(X_train)
    levels = np.array([check_support(r, ref)[0] for r in X_test.to_dict('records')])
    by_support = {}
    for lv in ('ok', 'edge', 'out'):
        m = levels == lv
        if m.any():
            by_support[lv] = {'n': int(m.sum()), **metrics(y_test.to_numpy()[m], pred_test[m])}
    print('[test] by support:', by_support)

    # Conformal fitted on train OOF only; untouched test measures actual coverage.
    print('[intervals] Out-of-fold calibration...')
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
    print('[intervals] coverage:', coverage)

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
    print('[importance] permutation on heldout test (descriptive only)')
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
        'model_name': best_name, 'price_unit': 'tỷ VND', 'features': FEATURES,
        'n_train': len(X_train), 'n_test': len(X_test),
        'train_metrics': train_m, 'test_metrics': test_m,
        'baseline_test_metrics': base_m, 'error_by_support': by_support,
        'tiny_backtest': {'internal_holdout': tiny_internal, 'external': external},
        'conformal': {**calib, 'test_coverage_pct': coverage,
                      'test_coverage80_by_support_pct': cov_support,
                      'test_coverage80_by_price_band_pct': cov_band,
                      'median_width_ratio_80': float(np.median(high / low))},
        'versions': {'python': platform.python_version(),
                     'sklearn': sklearn.__version__, 'catboost': catboost_version,
                     'pandas': pd.__version__, 'numpy': np.__version__}}
    if isinstance(final, StackedHouseRegressor):
        meta['stacking_ridge'] = {'coef': final.meta_.coef_.tolist(),
                                  'intercept': float(final.meta_.intercept_)}
    (MODEL_DIR / 'meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', round(time.time() - start), 'seconds; reports/ and models/ saved')


if __name__ == '__main__':
    main()
