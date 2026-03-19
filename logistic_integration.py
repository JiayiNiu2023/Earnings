"""
Pipeline Integration for Logistic Regression Module
=====================================================
Adds backtest_logistic() to the existing Pipeline class from pipeline_v6.py.

USAGE:
    from pipeline_v6 import *
    from logistic_module import *
    from logistic_integration import patch_pipeline

    # Patch the Pipeline class with logistic methods
    patch_pipeline()

    # Then use as normal:
    pipe = Pipeline(df, target="beta_adj_return", period_col="cal_quarter",
                    meta_cols=meta, exclude_cols=exclude)
    pipe.run(use_curated=True)
    pipe.backtest_logistic(prob_threshold=0.55, kelly_fraction=0.5, long_only=True)

LEAKAGE AUDIT CHECKLIST:
    ✓ Features are rank-transformed using ONLY training data
    ✓ Regularisation C is selected via purged time-series CV (train only)
    ✓ Model is fit on train, predictions on test — strict expanding window
    ✓ No feature uses future data (_is_lookahead filter inherited)
    ✓ Threshold sweep is reported as sensitivity analysis, not optimised
    ✓ All statistical tests use only OOS predictions
    ✓ Sector dummies derived from training categories only
"""

import numpy as np
import pandas as pd

from logistic_module import (
    compute_logistic_signal,
    compute_logistic_validation,
    run_logistic_backtest,
    sweep_thresholds,
    compute_calibration_table,
    brier_decomposition,
    hosmer_lemeshow_test,
    compute_delong_auc_ci,
    permutation_test_hit_rate,
    plot_calibration,
    plot_oos_diagnostics,
    plot_threshold_sweep,
    plot_feature_importance,
    plot_logistic_summary_table,
)

# We need these from pipeline_v6 — import at function time to avoid circular
# imports when the user loads things in different orders.


def _backtest_logistic(self, prob_threshold=0.55, kelly_fraction=0.5,
                        long_only=True, capital=1_000_000, cost_bps=10,
                        max_weight=0.15, min_train=4, purge=1,
                        n_cv_splits=3, sector_col=None,
                        sweep=True, plot=True):
    """
    Logistic regression backtest with full statistical validation.

    Walk-forward procedure (ZERO LEAKAGE):
    ──────────────────────────────────────
    For each test quarter t (t = min_train, min_train+1, ...):
      1. Training data  = all quarters [0, 1, ..., t-1]
      2. Rank-transform features using ONLY training distribution
      3. Select regularisation C via purged time-series CV within training
      4. Fit L1-logistic regression on full training set at best C
      5. Apply training-fitted rank-transform to quarter t
      6. Predict P(positive return) for quarter t
      7. Trade where P > prob_threshold (LONG) or P < 1-threshold (SHORT)

    Statistical validation (all OOS):
    ──────────────────────────────────
      - Brier score decomposition (reliability + resolution)
      - Hosmer-Lemeshow calibration test
      - DeLong AUC with 95% confidence interval
      - Permutation test for hit rate significance (5000 iterations)
      - Log-loss vs null model comparison
      - Per-period OOS metrics (AUC, Brier, hit rate)
      - Feature importance via L1 coefficient stability

    Parameters
    ----------
    prob_threshold : float
        Minimum P(up) to trigger a LONG trade. Default 0.55.
        Higher = fewer trades but higher accuracy.
    kelly_fraction : float
        Fraction of Kelly criterion for sizing. Default 0.5 (half-Kelly).
        0.5 is standard in practice — full Kelly is too aggressive.
    long_only : bool
        If True, only LONG trades. Default True.
    capital : float
        Starting capital. Default 1M.
    cost_bps : int
        Transaction cost in basis points. Default 10.
    max_weight : float
        Maximum per-name weight. Default 0.15.
    min_train : int
        Minimum training quarters before first prediction. Default 4.
    purge : int
        Purge gap for time-series CV (quarters). Default 1.
    n_cv_splits : int
        Number of CV folds for C selection. Default 3.
    sector_col : str or None
        Column for sector one-hot encoding. Default None.
    sweep : bool
        If True, run threshold sensitivity analysis. Default True.
    plot : bool
        If True, generate all diagnostic plots. Default True.

    Returns
    -------
    self (for chaining)
    """
    # Import from pipeline_v6 at runtime to avoid circular imports
    from pipeline_v6 import (compute_trade_stats, compute_date_returns,
                              plot_bt_equity, plot_bt_quarterly, plot_bt_summary,
                              plot_trade_stats, plot_trade_stats_table,
                              _print_bt_summary)

    print("\n" + "=" * 64)
    print("  LOGISTIC REGRESSION BACKTEST (v6)")
    print("=" * 64)
    print(f"  Mode: {'LONG ONLY' if long_only else 'LONG/SHORT'}")
    print(f"  P threshold: {prob_threshold}")
    print(f"  Kelly fraction: {kelly_fraction}")
    print(f"  Min training quarters: {min_train}")
    print(f"  Purge gap: {purge} quarter(s)")
    print(f"  CV folds: {n_cv_splits}")
    print(f"  Sector encoding: {sector_col or 'None'}")

    # ── Select features ──
    feats = (self.screened_features
             if hasattr(self, 'screened_features') and self.screened_features
             else self.curated_features)
    print(f"\n  Features ({len(feats)}): {feats}")

    # ── Verify no lookahead ──
    from pipeline_v6 import _is_lookahead
    lookahead_check = [f for f in feats if _is_lookahead(f)]
    if lookahead_check:
        print(f"\n  ⚠ WARNING: Lookahead features detected and removed: {lookahead_check}")
        feats = [f for f in feats if not _is_lookahead(f)]

    # ── Run walk-forward logistic ──
    print(f"\n  [1/5] Walk-forward logistic regression...")
    self.df["_log_prob"], self.logistic_diagnostics = compute_logistic_signal(
        self.df, feats, self.target, self.period_col,
        min_train=min_train, purge=purge, n_cv_splits=n_cv_splits,
        sector_col=sector_col
    )

    n_signal = self.df["_log_prob"].notna().sum()
    print(f"        Signal for {n_signal}/{len(self.df)} obs")

    if n_signal == 0:
        print("  No signal generated. Check min_train and data availability.")
        return self

    # ── Print per-period summary ──
    if self.logistic_diagnostics:
        print(f"\n  Per-period OOS summary ({len(self.logistic_diagnostics)} periods):")
        df_diag = pd.DataFrame(self.logistic_diagnostics)
        print(f"    AUC:    mean={df_diag['oos_auc'].mean():.3f}  "
              f"min={df_diag['oos_auc'].min():.3f}  max={df_diag['oos_auc'].max():.3f}")
        print(f"    Brier:  mean={df_diag['oos_brier'].mean():.4f}")
        print(f"    Active features: mean={df_diag['n_active_features'].mean():.1f}  "
              f"(L1 zeroed {len(feats) - df_diag['n_active_features'].mean():.1f} on average)")
        print(f"    Regularisation C: median={df_diag['best_C'].median():.4f}")

        # Feature selection stability
        all_active = {}
        for d in self.logistic_diagnostics:
            for f in d.get("active_features", []):
                all_active[f] = all_active.get(f, 0) + 1
        n_periods = len(self.logistic_diagnostics)
        print(f"\n  Feature selection stability (across {n_periods} OOS periods):")
        for f, count in sorted(all_active.items(), key=lambda x: -x[1])[:10]:
            print(f"    {f:40s}  selected {count}/{n_periods} ({count/n_periods:.0%})")

    # ── Run backtest ──
    print(f"\n  [2/5] Running backtest (threshold={prob_threshold}, "
          f"kelly={kelly_fraction})...")
    (self.logistic_trades, _,
     self.logistic_quarterly, self.logistic_equity) = run_logistic_backtest(
        self.df, "_log_prob", self.target, self.period_col,
        "announcement_date", prob_threshold=prob_threshold,
        kelly_fraction=kelly_fraction, long_only=long_only,
        initial_capital=capital, cost_bps=cost_bps, max_weight=max_weight
    )

    _print_bt_summary(self.logistic_trades, self.logistic_quarterly,
                      self.logistic_equity, capital, "Logistic", cost_bps)

    # ── Compute trade stats ──
    self.logistic_trade_stats = {}
    self.logistic_date_returns = {}
    if len(self.logistic_trades) > 0:
        for strat in self.logistic_trades["strategy"].unique():
            st = self.logistic_trades[self.logistic_trades["strategy"] == strat]
            self.logistic_trade_stats[strat] = compute_trade_stats(st, cost_bps)
            self.logistic_date_returns[strat] = compute_date_returns(st)

    # ── Statistical validation (all OOS) ──
    print(f"\n  [3/5] Statistical validation...")
    mask = self.df["_log_prob"].notna() & self.df[self.target].notna()
    y_true = (self.df.loc[mask, self.target] > 0).astype(int).values
    y_prob = self.df.loc[mask, "_log_prob"].values
    target_vals = self.df.loc[mask, self.target].values

    # Direction array (where we actually trade)
    y_dir = np.zeros(mask.sum())
    if "_log_prob" in self.df.columns:
        probs_masked = self.df.loc[mask, "_log_prob"].values
        y_dir[probs_masked > prob_threshold] = 1
        if not long_only:
            y_dir[probs_masked < (1 - prob_threshold)] = -1

    self.logistic_validation = compute_logistic_validation(
        y_true, y_prob, y_dir, target_vals, cost_bps=cost_bps
    )

    # Print key validation results
    v = self.logistic_validation
    b = v.get("brier", {})
    a = v.get("auc", {})
    hl = v.get("hosmer_lemeshow", {})
    pt = v.get("permutation_test", {})
    ll = v.get("logloss", {})

    print(f"\n  ── Statistical Validation (all OOS) ──")
    print(f"    AUC:           {a.get('auc', np.nan):.3f}  "
          f"[{a.get('ci_95_lower', np.nan):.3f}, {a.get('ci_95_upper', np.nan):.3f}] (95% DeLong CI)")
    print(f"    Brier Score:   {b.get('brier_score', np.nan):.4f}  "
          f"(skill={b.get('skill_score', 0):.3f} vs null)")
    print(f"      Reliability: {b.get('reliability', np.nan):.4f}  (calibration error, lower=better)")
    print(f"      Resolution:  {b.get('resolution', np.nan):.4f}  (discrimination, higher=better)")
    print(f"    H-L test:      p={hl.get('p_value', np.nan):.3f}  "
          f"({'well-calibrated' if hl.get('well_calibrated') else 'MISCALIBRATED'})")
    print(f"    Log-loss:      {ll.get('model', np.nan):.4f}  "
          f"(null={ll.get('null_model', np.nan):.4f}, "
          f"improvement={ll.get('improvement_pct', 0):.1f}%)")
    print(f"    Hit rate:      {pt.get('observed_hit_rate', np.nan):.3f}  "
          f"(permutation p={pt.get('p_value', np.nan):.3f}, "
          f"{'SIGNIFICANT' if pt.get('significant_at_05') else 'not significant'})")

    # ── Threshold sweep ──
    self.logistic_sweep = pd.DataFrame()
    if sweep:
        print(f"\n  [4/5] Threshold sensitivity analysis...")
        self.logistic_sweep = sweep_thresholds(
            self.df, "_log_prob", self.target, self.period_col,
            thresholds=[0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70],
            long_only=long_only, cost_bps=cost_bps,
            initial_capital=capital, kelly_fraction=kelly_fraction
        )
        if len(self.logistic_sweep) > 0:
            print(f"\n  ── Threshold Sweep ──")
            with pd.option_context("display.float_format", "{:.3f}".format,
                                   "display.max_rows", 20, "display.width", 140):
                print(self.logistic_sweep.to_string(index=False))
    else:
        print(f"\n  [4/5] Threshold sweep skipped.")

    # ── Plots ──
    if plot:
        print(f"\n  [5/5] Plotting...")

        # Calibration
        cal_table = compute_calibration_table(y_true, y_prob)
        if len(cal_table) > 0:
            plot_calibration(cal_table, label="Logistic OOS")

        # OOS diagnostics per period
        plot_oos_diagnostics(self.logistic_diagnostics, label="Walk-Forward")

        # Feature importance
        plot_feature_importance(self.logistic_diagnostics, feats)

        # Threshold sweep
        if sweep and len(self.logistic_sweep) > 0:
            plot_threshold_sweep(self.logistic_sweep)

        # Summary table
        plot_logistic_summary_table(
            self.logistic_validation, self.logistic_diagnostics,
            self.logistic_sweep, label="Logistic v6"
        )

        # Standard backtest plots (reuse from pipeline)
        if len(self.logistic_trades) > 0:
            from pipeline_v6 import (plot_bt_equity, plot_bt_quarterly,
                                      plot_bt_summary, plot_trade_stats,
                                      plot_trade_stats_table)
            plot_bt_equity(self.logistic_equity, capital)
            plot_bt_quarterly(self.logistic_quarterly)
            plot_bt_summary(self.logistic_quarterly, self.logistic_equity, capital)
            for strat in self.logistic_trade_stats:
                plot_trade_stats(self.logistic_trade_stats[strat],
                                self.logistic_date_returns[strat], strat)
                plot_trade_stats_table(self.logistic_trade_stats[strat],
                                      self.logistic_date_returns[strat], strat)
    else:
        print(f"\n  [5/5] Plotting skipped.")

    # ── Head-to-head vs ranking ──
    if (hasattr(self, 'ranking_trades') and self.ranking_trades is not None
            and len(self.ranking_trades) > 0 and len(self.logistic_trades) > 0):
        print(f"\n  ── Head-to-Head: Logistic vs Ranking ──")
        lo = ld = None
        for strat in self.logistic_trade_stats:
            ls = self.logistic_trade_stats[strat]
            lo = ls["overall"].iloc[0].to_dict() if len(ls["overall"]) > 0 else None
            ld = self.logistic_date_returns[strat]

        rank_strat = list(self.ranking_trade_stats.keys())[0] if self.ranking_trade_stats else None
        if rank_strat:
            rs = self.ranking_trade_stats[rank_strat]
            ro = rs["overall"].iloc[0].to_dict() if len(rs["overall"]) > 0 else None
            rd = self.ranking_date_returns[rank_strat]

            print(f"    {'Metric':<25s}  {'Ranking':>12s}  {'Logistic':>12s}")
            print(f"    {'─'*25}  {'─'*12}  {'─'*12}")

            def _fmt(val, fmt=".3f"):
                try:
                    return f"{val:{fmt}}" if np.isfinite(val) else "N/A"
                except (TypeError, ValueError):
                    return "N/A"

            if lo is not None and ro is not None:
                print(f"    {'Trades':<25s}  {ro.get('n_trades', 0):>12.0f}  {lo.get('n_trades', 0):>12.0f}")
                print(f"    {'Hit rate':<25s}  {_fmt(ro.get('hit_rate', np.nan), '.1%'):>12s}  {_fmt(lo.get('hit_rate', np.nan), '.1%'):>12s}")
                print(f"    {'Avg return (net)':<25s}  {_fmt(ro.get('avg_return_net', np.nan)):>12s}%  {_fmt(lo.get('avg_return_net', np.nan)):>12s}%")
                print(f"    {'Profit factor':<25s}  {_fmt(ro.get('profit_factor', np.nan), '.2f'):>12s}  {_fmt(lo.get('profit_factor', np.nan), '.2f'):>12s}")
                print(f"    {'Win/loss ratio':<25s}  {_fmt(ro.get('win_loss_ratio', np.nan), '.2f'):>12s}  {_fmt(lo.get('win_loss_ratio', np.nan), '.2f'):>12s}")

            if ld is not None and rd is not None and len(rd) > 1 and len(ld) > 1:
                r_sharpe = rd["portfolio_return"].mean() / rd["portfolio_return"].std() * np.sqrt(min(len(rd), 250))
                l_sharpe = ld["portfolio_return"].mean() / ld["portfolio_return"].std() * np.sqrt(min(len(ld), 250))
                print(f"    {'Date-level Sharpe':<25s}  {r_sharpe:>12.2f}  {l_sharpe:>12.2f}")

            r_final = self.ranking_equity["capital"].iloc[-1] if len(self.ranking_equity) > 0 else capital
            l_final = self.logistic_equity["capital"].iloc[-1] if len(self.logistic_equity) > 0 else capital
            print(f"    {'Final capital':<25s}  ${r_final:>11,.0f}  ${l_final:>11,.0f}")

    print("\n" + "=" * 64)
    print("  LOGISTIC BACKTEST COMPLETE")
    print("=" * 64)
    return self


def patch_pipeline():
    """
    Add backtest_logistic() method to the Pipeline class.

    Call this once after importing both pipeline_v6 and logistic_module.
    Idempotent — safe to call multiple times.
    """
    from pipeline_v6 import Pipeline

    Pipeline.backtest_logistic = _backtest_logistic

    # Also add storage attributes to __init__
    original_init = Pipeline.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.logistic_trades = None
        self.logistic_quarterly = None
        self.logistic_equity = None
        self.logistic_diagnostics = None
        self.logistic_validation = None
        self.logistic_trade_stats = {}
        self.logistic_date_returns = {}
        self.logistic_sweep = None

    Pipeline.__init__ = patched_init
    print("  ✓ Pipeline patched with backtest_logistic()")