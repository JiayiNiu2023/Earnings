"""
Industry Contagion Signal Pipeline
=====================================
Computes multi-metric industry signals from early reporters within each
quarter, measures their correlation with later reporters' price direction,
learns optimal weights walk-forward, and backtests.

Core idea:
  Within each industry-quarter, companies report on staggered dates.
  Early reporters reveal fundamental outcomes (EPS beat, revenue surprise, etc.)
  AND price reactions. Before the next company announces, we already know:
    - "6 out of 8 banks beat on EPS so far"
    - "Average revenue surprise in semis is +3.2%"
    - "70% of early-reporting SW stocks went up"
  
  Each of these is a potential signal for the NEXT reporter's price direction.
  We measure which signals actually predict price direction walk-forward,
  weight them by predictive power, and combine into a composite.

USAGE:
    from industry_contagion import IndustryContagionPipeline
    
    df = pd.read_csv("df.csv")
    
    pipe = IndustryContagionPipeline(df)
    pipe.compute_signals()              # Step 1: running industry stats
    pipe.measure_correlations()          # Step 2: which signals predict price?
    pipe.compute_composite()             # Step 3: weighted composite
    pipe.backtest(prob_threshold=0.55)   # Step 4: trade on it

LEAKAGE AUDIT:
  ✓ Each company only sees metrics from companies that reported BEFORE it
  ✓ Companies reporting on the SAME date are excluded (simultaneous)
  ✓ Correlation weights learned walk-forward (train on past quarters)
  ✓ First reporter in each industry gets NaN (no prior information)
  ✓ Metrics use only ACT vs EST (known at announcement time)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.special import expit
import warnings
warnings.filterwarnings("ignore")

POS, NEG, NEUT = "#2E86AB", "#D7263D", "#888"
PAL = ["#2E86AB", "#D7263D", "#F4A261", "#2A9D8F", "#7B2D8E",
       "#E76F51", "#264653", "#E9C46A"]

def _show(fig):
    fig.tight_layout(); plt.show(); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 1. METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

class Metric:
    """
    A metric computes a value from a single company's earnings data.
    It produces two things:
      - beat: 1 if positive surprise, 0 if negative, NaN if unavailable
      - surprise_pct: (ACT - EST) / |EST| * 100, clipped
    """
    def __init__(self, name, act_col, est_col, label=None, clip=(-200, 200)):
        self.name = name
        self.act_col = act_col
        self.est_col = est_col
        self.label = label or name
        self.clip_lo, self.clip_hi = clip

    def compute(self, row):
        """Returns (beat, surprise_pct) for a single row."""
        act = row.get(self.act_col, np.nan)
        est = row.get(self.est_col, np.nan)
        if pd.isna(act) or pd.isna(est) or abs(est) < 1e-10:
            return np.nan, np.nan
        surprise_pct = (act - est) / abs(est) * 100
        surprise_pct = np.clip(surprise_pct, self.clip_lo, self.clip_hi)
        beat = 1.0 if surprise_pct > 0 else 0.0
        return beat, surprise_pct

    def compute_series(self, df):
        """Vectorised computation for a DataFrame."""
        act = df[self.act_col].astype(float)
        est = df[self.est_col].astype(float)
        valid = act.notna() & est.notna() & (est.abs() > 1e-10)
        surprise = pd.Series(np.nan, index=df.index)
        surprise[valid] = ((act[valid] - est[valid]) / est[valid].abs() * 100).clip(
            self.clip_lo, self.clip_hi)
        beat = pd.Series(np.nan, index=df.index)
        beat[valid] = (surprise[valid] > 0).astype(float)
        return beat, surprise


def get_default_metrics(df_columns):
    """
    Define the default set of metrics from ACT/EST pairs.
    Only include pairs where both columns exist in the data.
    """
    METRIC_DEFS = [
        ("EPS",      "ACT_9_EARNINGS_PER_SHARE",     "EST_9_EARNINGS_PER_SHARE"),
        ("Revenue",  "ACT_20_SALES",                  "EST_20_SALES"),
        ("GrossMargin", "ACT_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN",
                        "EST_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN"),
        ("EBIT",     "ACT_6_EARNINGS_BEFORE_INTEREST_AND_TAXES",
                     "EST_6_EARNINGS_BEFORE_INTEREST_AND_TAXES"),
        ("EBITDA",   "ACT_8_EARNINGS_BEFORE_INTEREST_TAXES_AND_DEPRECIATION",
                     "EST_8_EARNINGS_BEFORE_INTEREST_TAXES_AND_DEPRECIATION"),
        ("NetIncome","ACT_15_NET_INCOME",             "EST_15_NET_INCOME"),
        ("FCF",      "ACT_237_FREE_CASH_FLOW",        "EST_237_FREE_CASH_FLOW"),
        ("OpEx",     "ACT_104_OPERATING_EXPENSE",      "EST_104_OPERATING_EXPENSE"),
        ("CFPS",     "ACT_2_CASH_FLOW_PER_SHARE",     "EST_2_CASH_FLOW_PER_SHARE"),
        ("BV",       "ACT_1_BOOK_VALUE_PER_SHARE",     "EST_1_BOOK_VALUE_PER_SHARE"),
    ]
    available = set(df_columns)
    metrics = []
    for name, act, est in METRIC_DEFS:
        if act in available and est in available:
            metrics.append(Metric(name, act, est))
    return metrics


# ═══════════════════════════════════════════════════════════════════
# 2. RUNNING INDUSTRY STATISTICS
# ═══════════════════════════════════════════════════════════════════

def compute_industry_signals(df, metrics, industry_col="industry",
                              quarter_col="cal_quarter",
                              date_col="announcement_date",
                              price_col="price_change",
                              target_col="beta_adj_return",
                              min_prior=1):
    """
    For each company, compute running industry statistics from all
    companies in the SAME industry-quarter that reported STRICTLY BEFORE it.

    Produces per-company columns:
      ind_n_prior          : how many in the industry reported before me
      ind_n_total          : total in the industry this quarter
      ind_progress         : n_prior / n_total
      ind_price_pos_rate   : fraction of prior reporters with positive price_change
      ind_avg_price_change : mean price change of prior reporters
      ind_{metric}_beat_rate    : fraction of prior reporters that beat on {metric}
      ind_{metric}_avg_surprise : mean surprise % of prior reporters on {metric}
      ind_{metric}_pos_price_when_beat : P(price > 0 | beat) from prior reporters

    The last one is the KEY innovation: it's the conditional probability that
    a stock goes up given that it beat on this metric, estimated from
    early reporters in the same industry this quarter.

    Parameters
    ----------
    min_prior : int
        Minimum number of prior reporters needed before computing signals.
        Default 1 (need at least 1 prior reporter).

    Returns
    -------
    df with new columns added. First reporter in each industry gets NaN.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([quarter_col, industry_col, date_col, "ticker"]).reset_index(drop=True)

    # Pre-compute beats and surprises for all metrics
    beat_cols = {}
    surp_cols = {}
    for m in metrics:
        bcol = f"_beat_{m.name}"
        scol = f"_surp_{m.name}"
        df[bcol], df[scol] = m.compute_series(df)
        beat_cols[m.name] = bcol
        surp_cols[m.name] = scol

    # Positive price indicator
    df["_price_pos"] = (df[price_col] > 0).astype(float)

    # ── Output columns ──
    new_cols = {
        "ind_n_prior": np.nan,
        "ind_n_total": np.nan,
        "ind_progress": np.nan,
        "ind_price_pos_rate": np.nan,
        "ind_avg_price_change": np.nan,
        "ind_price_vol": np.nan,
    }
    for m in metrics:
        new_cols[f"ind_{m.name}_beat_rate"] = np.nan
        new_cols[f"ind_{m.name}_avg_surprise"] = np.nan
        new_cols[f"ind_{m.name}_pos_price_when_beat"] = np.nan
        new_cols[f"ind_{m.name}_pos_price_when_miss"] = np.nan

    for col, val in new_cols.items():
        df[col] = val

    # ── Process each industry-quarter ──
    n_computed = 0
    n_skipped = 0

    for (quarter, industry), grp in df.groupby([quarter_col, industry_col]):
        if len(grp) < 2:
            n_skipped += len(grp)
            continue

        grp_sorted = grp.sort_values([date_col, "ticker"])
        indices = grp_sorted.index.tolist()
        dates = grp_sorted[date_col].values
        n_total = len(grp_sorted)

        for i, idx in enumerate(indices):
            current_date = dates[i]
            # STRICTLY before: prior reporters are those with date < current_date
            prior_mask = dates[:i + 1] < current_date  # includes positions 0..i
            # But we also need to handle same-date: exclude companies reporting
            # on the same date (simultaneous reporters can't see each other)
            prior_indices = [indices[j] for j in range(i + 1) if dates[j] < current_date]

            n_prior = len(prior_indices)
            df.at[idx, "ind_n_prior"] = n_prior
            df.at[idx, "ind_n_total"] = n_total
            df.at[idx, "ind_progress"] = n_prior / n_total

            if n_prior < min_prior:
                n_skipped += 1
                continue

            prior = df.loc[prior_indices]

            # Price-based signals
            pp = prior["_price_pos"]
            pc = prior[price_col]
            df.at[idx, "ind_price_pos_rate"] = pp.mean()
            df.at[idx, "ind_avg_price_change"] = pc.mean()
            df.at[idx, "ind_price_vol"] = pc.std() if len(pc) > 1 else np.nan

            # Metric-based signals
            for m in metrics:
                bcol = beat_cols[m.name]
                scol = surp_cols[m.name]
                beats = prior[bcol].dropna()
                surps = prior[scol].dropna()

                if len(beats) > 0:
                    df.at[idx, f"ind_{m.name}_beat_rate"] = beats.mean()
                if len(surps) > 0:
                    df.at[idx, f"ind_{m.name}_avg_surprise"] = surps.mean()

                # Conditional probability: P(price > 0 | beat) from prior reporters
                if len(beats) > 0:
                    beat_mask = prior[bcol] == 1.0
                    miss_mask = prior[bcol] == 0.0
                    if beat_mask.sum() > 0:
                        df.at[idx, f"ind_{m.name}_pos_price_when_beat"] = \
                            prior.loc[beat_mask, "_price_pos"].mean()
                    if miss_mask.sum() > 0:
                        df.at[idx, f"ind_{m.name}_pos_price_when_miss"] = \
                            prior.loc[miss_mask, "_price_pos"].mean()

            n_computed += 1

    # Clean up temp columns
    for m in metrics:
        df.drop(columns=[beat_cols[m.name], surp_cols[m.name]], inplace=True)
    df.drop(columns=["_price_pos"], inplace=True)

    # Report
    signal_cols = [c for c in df.columns if c.startswith("ind_")]
    print(f"\n  Industry Contagion Signals Computed:")
    print(f"    {n_computed} observations with signals, {n_skipped} skipped (first/too few)")
    print(f"    {len(signal_cols)} signal columns created")
    for c in signal_cols:
        nn = df[c].notna().sum()
        if nn > 0:
            print(f"      {c:45s}  {nn:>4d} non-null  mean={df[c].mean():.4f}")

    return df, signal_cols


# ═══════════════════════════════════════════════════════════════════
# 3. CORRELATION MEASUREMENT — which signals predict price?
# ═══════════════════════════════════════════════════════════════════

def measure_signal_correlations(df, signal_cols, target_col="beta_adj_return",
                                 quarter_col="cal_quarter"):
    """
    Per-quarter Spearman correlation between each signal and the target.
    This tells us: does a higher EPS beat rate among prior reporters
    predict higher returns for the next reporter?

    Returns
    -------
    panel : DataFrame (periods × signals) of per-period correlations
    summary : DataFrame with mean, std, IR, hit_rate per signal
    """
    periods = sorted(df[quarter_col].unique())
    recs = []
    for p in periods:
        m = df[quarter_col] == p
        y = df.loc[m, target_col]
        row = {"period": p}
        for s in signal_cols:
            x = df.loc[m, s]
            v = x.notna() & y.notna()
            if v.sum() < 8:
                row[s] = np.nan; continue
            r, _ = spearmanr(x[v], y[v])
            row[s] = r if not np.isnan(r) else np.nan
        recs.append(row)

    panel = pd.DataFrame(recs).set_index("period")

    # Summary statistics
    rows = []
    for s in signal_cols:
        ts = panel[s].dropna()
        if len(ts) < 3: continue
        m, sd = ts.mean(), ts.std(ddof=1)
        ir = m / sd if sd > 1e-12 else 0
        hit = (ts * np.sign(m) > 0).mean() if abs(m) > 1e-12 else 0.5
        rows.append({"signal": s, "mean_ic": m, "std_ic": sd, "ir": ir,
                     "hit_rate": hit, "n_periods": len(ts)})

    summary = pd.DataFrame(rows).sort_values("ir", ascending=False, key=abs).reset_index(drop=True)
    return panel, summary


def measure_conditional_hit_rates(df, signal_cols, target_col="beta_adj_return",
                                   quarter_col="cal_quarter"):
    """
    For each signal, compute the hit rate when signal is in top vs bottom quartile.
    This is a more direct measure: "when the industry EPS beat rate is high,
    how often does the next reporter go up?"
    """
    rows = []
    for s in signal_cols:
        top_hits, bot_hits = [], []
        for p in df[quarter_col].unique():
            m = df[quarter_col] == p
            x, y = df.loc[m, s], df.loc[m, target_col]
            v = x.notna() & y.notna()
            if v.sum() < 8: continue
            xv, yv = x[v], y[v]
            q75, q25 = xv.quantile(0.75), xv.quantile(0.25)
            top = yv[xv >= q75]; bot = yv[xv <= q25]
            if len(top) > 0: top_hits.extend((top > 0).astype(float).tolist())
            if len(bot) > 0: bot_hits.extend((bot > 0).astype(float).tolist())

        if len(top_hits) < 5 or len(bot_hits) < 5: continue
        rows.append({
            "signal": s,
            "top_q_hit_rate": np.mean(top_hits),
            "bot_q_hit_rate": np.mean(bot_hits),
            "hit_spread": np.mean(top_hits) - np.mean(bot_hits),
            "top_n": len(top_hits), "bot_n": len(bot_hits),
        })

    return pd.DataFrame(rows).sort_values("hit_spread", ascending=False, key=abs).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD COMPOSITE SIGNAL
# ═══════════════════════════════════════════════════════════════════

def compute_composite_signal(df, signal_cols, target_col="beta_adj_return",
                              quarter_col="cal_quarter",
                              min_train=4, icir_threshold=0.15,
                              max_signals=10):
    """
    Walk-forward ICIR-weighted composite of industry signals.

    For each test quarter t:
      1. Compute per-signal ICIR from quarters [0, t-1]
      2. Keep signals with |ICIR| > threshold
      3. Weight by ICIR, normalise
      4. Z-score signals within the quarter
      5. Composite = weighted sum of z-scores

    Returns
    -------
    composite : Series of composite scores (higher = more bullish)
    weights_history : DataFrame of signal weights per period
    """
    periods = sorted(df[quarter_col].unique())
    composite = pd.Series(np.nan, index=df.index)
    wt_recs = []

    for ti in range(min_train, len(periods)):
        test_period = periods[ti]
        train_periods = periods[:ti]

        # Compute per-signal IC from training data
        train_mask = df[quarter_col].isin(train_periods)
        train_df = df[train_mask]

        ic_panel, _ = measure_signal_correlations(
            train_df, signal_cols, target_col, quarter_col)

        # Select and weight signals
        weights = {}
        for s in signal_cols:
            if s not in ic_panel.columns: continue
            ts = ic_panel[s].dropna()
            if len(ts) < 3: continue
            m, sd = ts.mean(), ts.std(ddof=1)
            icir = m / sd if sd > 1e-12 else 0
            if abs(icir) > icir_threshold:
                weights[s] = icir

        if not weights:
            continue

        # Keep top N by |ICIR|
        if len(weights) > max_signals:
            sorted_w = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
            weights = dict(sorted_w[:max_signals])

        # Normalise
        total = sum(abs(w) for w in weights.values())
        if total < 1e-12: continue
        weights = {s: w / total for s, w in weights.items()}
        wt_recs.append({"period": test_period, **weights})

        # Apply to test period
        test_mask = df[quarter_col] == test_period
        test_idx = df.index[test_mask]

        sig = np.zeros(test_mask.sum())
        for s, w in weights.items():
            vals = df.loc[test_mask, s].values.astype(float)
            # Z-score within test period (cross-sectional)
            mu = np.nanmean(vals)
            sd = np.nanstd(vals)
            z = (vals - mu) / max(sd, 1e-12)
            z = np.nan_to_num(z, 0)
            sig += w * z

        composite.loc[test_idx] = sig

    df_wt = pd.DataFrame(wt_recs)
    if len(df_wt) > 0:
        df_wt = df_wt.set_index("period")
    return composite, df_wt


# ═══════════════════════════════════════════════════════════════════
# 5. BACKTEST
# ═══════════════════════════════════════════════════════════════════

def run_contagion_backtest(df, signal_col, target, pcol,
                           date_col="announcement_date",
                           initial_capital=1_000_000, cost_bps=10,
                           max_weight=0.15, long_only=True,
                           top_pct=0.20):
    """
    Rank-based backtest: go LONG on stocks in the top percentile
    of the composite signal each quarter.
    """
    df = df.copy()
    df = df[df[signal_col].notna() & df[target].notna()].copy()
    df = df.sort_values([pcol, date_col, "ticker"])

    # Per-quarter thresholds
    qt = {}
    for q in df[pcol].unique():
        qs = df.loc[df[pcol] == q, signal_col]
        if len(qs) < 4: continue
        qt[q] = {"top": qs.quantile(1 - top_pct), "bot": qs.quantile(top_pct)}

    df["_dir"] = 0
    df["_conv"] = 0.0
    for q, th in qt.items():
        mask = df[pcol] == q
        df.loc[mask & (df[signal_col] >= th["top"]), "_dir"] = 1
        if not long_only:
            df.loc[mask & (df[signal_col] <= th["bot"]), "_dir"] = -1
        df.loc[mask, "_conv"] = df.loc[mask, signal_col].abs()

    label = "contagion_long" if long_only else "contagion_ls"
    df["_conf_reasons"] = ""
    cost_frac = cost_bps / 10000
    trade_dates = sorted(df.loc[df["_dir"] != 0, date_col].unique())

    all_trades = []
    capital = initial_capital

    for tdate in trade_dates:
        batch = df[(df[date_col] == tdate) & (df["_dir"] != 0)]
        if len(batch) < 1: continue
        quarter = batch[pcol].iloc[0]
        dirs = batch["_dir"].values.astype(int)
        rets = batch[target].values
        tickers = batch["ticker"].values
        n = len(batch)

        weights = np.ones(n) / n
        weights = np.minimum(weights, max_weight)

        gross_rets = dirs * rets / 100
        net_rets = gross_rets - cost_frac
        pnls = capital * weights * net_rets

        for j in range(n):
            all_trades.append(dict(
                strategy=label, date=tdate, quarter=quarter,
                ticker=tickers[j], direction=int(dirs[j]),
                conviction=0.5, weight=round(float(weights[j]), 6),
                capital_pre=round(capital, 2),
                return_pct=round(rets[j], 4),
                gross_pnl=round(capital * weights[j] * dirs[j] * rets[j] / 100, 2),
                cost=round(capital * weights[j] * cost_frac, 2),
                net_pnl=round(pnls[j], 2),
                hit=int(dirs[j] * rets[j] > 0),
                _conf_reasons="",
            ))
        capital += pnls.sum()

    df_trades = pd.DataFrame(all_trades)
    if len(df_trades) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Aggregate
    df_quarterly = df_trades.groupby(["strategy", "quarter"]).agg(
        n_trades=("net_pnl", "count"),
        n_long=("direction", lambda x: (x > 0).sum()),
        gross_pnl=("gross_pnl", "sum"), total_cost=("cost", "sum"),
        net_pnl=("net_pnl", "sum"), hit_rate=("hit", "mean"),
    ).reset_index()

    cum_cap = initial_capital; ret_pcts = []
    for _, row in df_quarterly.iterrows():
        ret_pcts.append(row["net_pnl"] / cum_cap * 100); cum_cap += row["net_pnl"]
    df_quarterly["return_pct"] = ret_pcts

    eq_recs = [dict(strategy=label, date="start", capital=initial_capital)]
    cap = initial_capital
    for _, row in df_trades.sort_values("date").groupby("date").agg(net_pnl=("net_pnl", "sum")).iterrows():
        cap += row["net_pnl"]; eq_recs.append(dict(strategy=label, date=_, capital=cap))

    return df_trades, df_quarterly, pd.DataFrame(eq_recs)


# ═══════════════════════════════════════════════════════════════════
# 6. PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_signal_summary(summary, top_n=25):
    if len(summary) == 0: return
    d = summary.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(d) * 0.3)))
    y = np.arange(len(d)); feats = d["signal"].values
    colors = [POS if v > 0 else NEG for v in d["mean_ic"]]
    ax.barh(y, d["mean_ic"], height=0.6, color=colors, alpha=0.7)
    for i, row in d.iterrows():
        ax.annotate(f'IR={row["ir"]:.2f}  hit={row["hit_rate"]:.0%}',
                     xy=(row["mean_ic"], list(d.index).index(i)),
                     xytext=(4 if row["mean_ic"] >= 0 else -4, 0),
                     textcoords="offset points", fontsize=6,
                     ha="left" if row["mean_ic"] >= 0 else "right")
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=7)
    ax.axvline(0, c="black", lw=0.5); ax.invert_yaxis()
    ax.set_xlabel("Mean IC"); ax.set_title("Industry Signal Correlations with Price Direction")
    ax.grid(axis="x", alpha=0.3); _show(fig)


def plot_conditional_hits(cond_df, top_n=15):
    if len(cond_df) == 0: return
    d = cond_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, len(d) * 0.35)))
    y = np.arange(len(d)); w = 0.35
    ax.barh(y - w / 2, d["top_q_hit_rate"] * 100, w, color=POS, alpha=0.7, label="Top quartile")
    ax.barh(y + w / 2, d["bot_q_hit_rate"] * 100, w, color=NEG, alpha=0.7, label="Bot quartile")
    ax.axvline(50, c="black", lw=0.5, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(d["signal"].values, fontsize=7)
    ax.set_xlabel("Hit Rate (%)"); ax.set_title("Hit Rate: Top vs Bottom Quartile")
    ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    _show(fig)


def plot_weights_over_time(weights_df, top_n=8):
    if len(weights_df) == 0: return
    avg_wts = weights_df.mean().abs().sort_values(ascending=False)
    top_sigs = avg_wts.head(top_n).index.tolist()
    if not top_sigs: return

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, s in enumerate(top_sigs):
        if s in weights_df.columns:
            ts = weights_df[s].dropna()
            ax.plot(range(len(ts)), ts.values, color=PAL[i % len(PAL)], lw=1.2,
                    marker="o", ms=3, label=s.replace("ind_", ""))
    ax.axhline(0, c="black", lw=0.5)
    ax.set_xticks(range(len(weights_df)))
    ax.set_xticklabels(weights_df.index, rotation=45, fontsize=6)
    ax.set_ylabel("Weight"); ax.set_title("Signal Weights Over Time (Walk-Forward)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3); _show(fig)


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════

class IndustryContagionPipeline:
    """
    Complete pipeline: read data → compute industry signals → measure
    correlations → build composite → backtest.
    """

    def __init__(self, df, metrics=None, industry_col="industry",
                 quarter_col="cal_quarter", date_col="announcement_date",
                 price_col="price_change", target_col="beta_adj_return",
                 sp500_col="sp500_ret"):
        self.df = df.copy()
        self.industry_col = industry_col
        self.quarter_col = quarter_col
        self.date_col = date_col
        self.price_col = price_col
        self.target_col = target_col
        self.sp500_col = sp500_col

        # Ensure date is datetime
        self.df[date_col] = pd.to_datetime(self.df[date_col])

        # Metrics
        if metrics is None:
            self.metrics = get_default_metrics(self.df.columns)
        else:
            self.metrics = metrics

        print(f"  Industry Contagion Pipeline")
        print(f"    {len(self.df)} obs | {self.df[quarter_col].nunique()} quarters "
              f"| {self.df[industry_col].nunique()} industries")
        print(f"    Metrics ({len(self.metrics)}): {[m.name for m in self.metrics]}")

        # Storage
        self.signal_cols = []
        self.ic_panel = None
        self.ic_summary = None
        self.cond_hits = None
        self.composite = None
        self.composite_weights = None
        self.trades = None
        self.quarterly = None
        self.equity = None

    def add_metric(self, name, act_col, est_col, clip=(-200, 200)):
        """Add a custom metric."""
        self.metrics.append(Metric(name, act_col, est_col, clip=clip))
        print(f"    Added metric: {name} ({act_col} vs {est_col})")

    def compute_signals(self, min_prior=1):
        """Step 1: Compute running industry statistics from early reporters."""
        print(f"\n{'='*64}")
        print(f"  STEP 1: Computing Industry Signals")
        print(f"{'='*64}")
        self.df, self.signal_cols = compute_industry_signals(
            self.df, self.metrics, self.industry_col, self.quarter_col,
            self.date_col, self.price_col, self.target_col, min_prior)
        return self

    def measure_correlations(self, plot=True):
        """Step 2: Measure which signals predict price direction."""
        print(f"\n{'='*64}")
        print(f"  STEP 2: Measuring Signal-Price Correlations")
        print(f"{'='*64}")
        if not self.signal_cols:
            print("  No signals computed. Run compute_signals() first.")
            return self

        self.ic_panel, self.ic_summary = measure_signal_correlations(
            self.df, self.signal_cols, self.target_col, self.quarter_col)

        print(f"\n  ── Top Signals by ICIR ──")
        if len(self.ic_summary) > 0:
            for _, r in self.ic_summary.head(15).iterrows():
                sig = "***" if abs(r["ir"]) > 1.0 else "**" if abs(r["ir"]) > 0.5 else "*" if abs(r["ir"]) > 0.3 else ""
                print(f"    {r['signal']:45s}  IC={r['mean_ic']:+.4f}  IR={r['ir']:+.3f}  "
                      f"hit={r['hit_rate']:.0%}  {sig}")

        # Conditional hit rates
        self.cond_hits = measure_conditional_hit_rates(
            self.df, self.signal_cols, self.target_col, self.quarter_col)

        if len(self.cond_hits) > 0:
            print(f"\n  ── Top Signals by Conditional Hit Spread ──")
            for _, r in self.cond_hits.head(10).iterrows():
                print(f"    {r['signal']:45s}  top_hit={r['top_q_hit_rate']:.1%}  "
                      f"bot_hit={r['bot_q_hit_rate']:.1%}  spread={r['hit_spread']:+.1%}")

        if plot:
            plot_signal_summary(self.ic_summary)
            plot_conditional_hits(self.cond_hits)

        return self

    def compute_composite(self, min_train=4, icir_threshold=0.15,
                           max_signals=10, plot=True):
        """Step 3: Build walk-forward ICIR-weighted composite."""
        print(f"\n{'='*64}")
        print(f"  STEP 3: Walk-Forward Composite Signal")
        print(f"{'='*64}")
        print(f"    Min training quarters: {min_train}")
        print(f"    ICIR threshold: {icir_threshold}")
        print(f"    Max signals: {max_signals}")

        self.df["_contagion_composite"], self.composite_weights = \
            compute_composite_signal(
                self.df, self.signal_cols, self.target_col, self.quarter_col,
                min_train, icir_threshold, max_signals)

        n = self.df["_contagion_composite"].notna().sum()
        print(f"\n    Composite signal for {n}/{len(self.df)} obs")

        if self.composite_weights is not None and len(self.composite_weights) > 0:
            avg_wts = self.composite_weights.mean().abs().sort_values(ascending=False)
            print(f"\n  ── Average Signal Weights (top 10) ──")
            for s, w in avg_wts.head(10).items():
                print(f"    {s:45s}  |w|={w:.4f}")

            if plot:
                plot_weights_over_time(self.composite_weights)

        return self

    def backtest(self, capital=1_000_000, cost_bps=10, long_only=True,
                 top_pct=0.20, plot=True):
        """Step 4: Backtest the composite signal."""
        print(f"\n{'='*64}")
        print(f"  STEP 4: Backtest")
        print(f"{'='*64}")
        print(f"    Mode: {'LONG ONLY' if long_only else 'LONG/SHORT'}")
        print(f"    Top pct: {top_pct:.0%}")
        print(f"    Capital: ${capital:,.0f}")

        self.trades, self.quarterly, self.equity = run_contagion_backtest(
            self.df, "_contagion_composite", self.target_col, self.quarter_col,
            self.date_col, capital, cost_bps, 0.15, long_only, top_pct)

        if len(self.trades) == 0:
            print("  No trades generated.")
            return self

        # Summary
        final = self.equity["capital"].iloc[-1] if len(self.equity) > 0 else capital
        tr = (final / capital - 1) * 100
        t = self.trades
        t["trade_return"] = t["direction"] * t["return_pct"]
        t["trade_return_net"] = t["trade_return"] - cost_bps / 100
        wins = t[t["trade_return_net"] > 0]
        losses = t[t["trade_return_net"] <= 0]
        hr = t["hit"].mean()
        pf = wins["trade_return_net"].sum() / abs(losses["trade_return_net"].sum()) if len(losses) > 0 and losses["trade_return_net"].sum() != 0 else np.nan
        wl = abs(wins["trade_return_net"].mean() / losses["trade_return_net"].mean()) if len(wins) > 0 and len(losses) > 0 and losses["trade_return_net"].mean() != 0 else np.nan

        # Date-level Sharpe
        date_rets = t.groupby("date")["trade_return"].mean()
        sharpe = date_rets.mean() / date_rets.std() * np.sqrt(min(len(date_rets), 250)) if len(date_rets) > 1 and date_rets.std() > 0 else 0

        print(f"\n  ── Results ──")
        print(f"    Trades:        {len(t)}")
        print(f"    Hit rate:      {hr:.1%}")
        print(f"    Avg return:    {t['trade_return_net'].mean():.3f}%")
        print(f"    Win/loss:      {wl:.2f}")
        print(f"    Profit factor: {pf:.2f}")
        print(f"    Sharpe:        {sharpe:.2f}")
        print(f"    Final capital: ${final:,.0f} ({tr:+.1f}%)")

        if plot and len(self.quarterly) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

            # Quarterly PnL
            qs = self.quarterly
            ax1.bar(range(len(qs)), qs["net_pnl"], color=[POS if v > 0 else NEG for v in qs["net_pnl"]], alpha=0.7)
            ax1.set_xticks(range(len(qs)))
            ax1.set_xticklabels(qs["quarter"], rotation=45, fontsize=6)
            ax1.axhline(0, c="black", lw=0.5); ax1.set_ylabel("Net PnL ($)")
            ax1.set_title("Quarterly PnL"); ax1.grid(axis="y", alpha=0.3)

            # Hit rate over time
            q_hits = qs["hit_rate"]
            ax2.bar(range(len(qs)), q_hits * 100, color=[POS if v > 0.5 else NEG for v in q_hits], alpha=0.7)
            ax2.axhline(50, c="black", lw=0.5, ls="--")
            ax2.set_xticks(range(len(qs)))
            ax2.set_xticklabels(qs["quarter"], rotation=45, fontsize=6)
            ax2.set_ylabel("Hit Rate (%)"); ax2.set_title("Quarterly Hit Rate"); ax2.grid(axis="y", alpha=0.3)

            fig.suptitle("Industry Contagion Backtest", fontsize=12)
            _show(fig)

        print(f"\n{'='*64}")
        print(f"  CONTAGION PIPELINE COMPLETE")
        print(f"{'='*64}")
        return self

    def get_signals_df(self):
        """Return the DataFrame with all signals attached."""
        return self.df