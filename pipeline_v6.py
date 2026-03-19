"""
Earnings Signal Pipeline v6
============================
Major changes from v5:
  - Beta-adjusted target (rolling 60-day beta per ticker, not beta=1)
  - Orthogonal voter system (5 independent signals, no redundancy)
  - Long-only default (SHORT side validated separately before enabling)
  - Higher conviction threshold (min 4 net votes)
  - Curated feature list for ranking (no kitchen-sink)
  - Feature-level IC screening before inclusion

Two signal frameworks:
  1. RANKING: cross-sectional ICIR-weighted z-scores (curated features)
  2. CONFLUENCE: absolute directional voting — each signal votes LONG,
     trade only when enough votes agree

USAGE:
    from pipeline import *

    # With real data:
    df = pd.read_excel('earnings_df.xlsx')  # your processed DataFrame
    spx = pd.read_excel('spx_open_close.xlsx')  # SPX data

    # Beta-adjust the target (replaces naive price_change - sp500_ret)
    df = compute_beta_adjusted_target(df, spx)

    meta = ["ticker","name","sector","industry","sub_industry",
            "announcement_date","ann_type","cal_quarter", ...]
    exclude = ["sp500_ret","price_change","price_change_resid"]

    pipe = Pipeline(df, target="beta_adj_return", period_col="cal_quarter",
                    meta_cols=meta, exclude_cols=exclude)
    pipe.run()
    pipe.backtest_ranking(capital=1_000_000)
    pipe.backtest_confluence(min_votes=4, long_only=True, capital=1_000_000)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, t as t_dist, beta as beta_dist
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# TARGET CONSTRUCTION: Beta-adjusted returns
# ═══════════════════════════════════════════════════════════════════

def compute_beta_adjusted_target(df, spx_df=None, window=60, fallback_beta=1.0):
    """
    Compute beta-adjusted post-earnings return per stock.

    Instead of: price_change - sp500_ret  (assumes beta=1 for all stocks)
    Computes:   price_change - beta_i * sp500_ret

    Where beta_i is estimated from trailing `window` earning announcements
    for that ticker (regressing price_change on sp500_ret).

    If spx_df is None, assumes df already has 'sp500_ret' column.

    Parameters
    ----------
    df : DataFrame with 'ticker', 'price_change', 'sp500_ret', 'cal_quarter'
    spx_df : optional DataFrame for SPX data (unused if sp500_ret already in df)
    window : number of trailing observations per ticker for beta estimation
    fallback_beta : beta to use when insufficient history

    Returns
    -------
    df with new column 'beta_adj_return'
    """
    df = df.copy()
    df = df.sort_values(["ticker", "cal_quarter"]).reset_index(drop=True)

    # Estimate rolling beta per ticker
    betas = pd.Series(fallback_beta, index=df.index, dtype=float)

    for tk in df["ticker"].unique():
        mask = df["ticker"] == tk
        tk_df = df.loc[mask, ["price_change", "sp500_ret"]].dropna()

        if len(tk_df) < 4:
            continue

        # Expanding beta with minimum window
        for i in range(len(tk_df)):
            if i < 3:  # need at least 4 obs
                continue
            start = max(0, i - window)
            sub = tk_df.iloc[start:i]
            x = sub["sp500_ret"].values
            y = sub["price_change"].values

            if x.std() < 1e-8:
                continue

            # OLS beta: cov(y,x) / var(x)
            beta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
            beta = np.clip(beta, 0.2, 3.0)  # sanity bounds
            betas.loc[tk_df.index[i]] = beta

    df["_ticker_beta"] = betas
    df["beta_adj_return"] = df["price_change"] - df["_ticker_beta"] * df["sp500_ret"]

    # Also keep the naive version for comparison
    if "price_change_resid" not in df.columns:
        df["price_change_resid"] = df["price_change"] - df["sp500_ret"]

    print(f"  Beta-adjusted target: mean={df['beta_adj_return'].mean():.4f}  "
          f"std={df['beta_adj_return'].std():.4f}")
    print(f"  Ticker beta range: [{df['_ticker_beta'].min():.2f}, {df['_ticker_beta'].max():.2f}]  "
          f"median={df['_ticker_beta'].median():.2f}")
    return df


# ═══════════════════════════════════════════════════════════════════
# CURATED FEATURE LISTS
# ═══════════════════════════════════════════════════════════════════

def get_curated_features(df_columns):
    """
    Return a curated list of features for ranking, organized by economic theme.

    Philosophy: only include features where there is a clear economic reason
    to expect predictive power for post-earnings returns. Exclude:
      - Raw ACT_*/EST_* fundamentals (cross-sectionally incomparable)
      - Redundant variants of the same signal
      - Features with no economic thesis

    Each theme contributes 3-6 features. Total ~30, not 162.
    """
    CURATED = {
        # Theme 1: Analyst revision momentum (consensus is shifting)
        "revision": [
            "Rev_CS_Full",          # composite cross-sectional revision rank
            "RevZ_VolAdj",          # revision z-score adjusted for analyst disagreement
            "Rev_Sign_Consensus",   # do 3d/7d/14d revisions all agree on direction?
            "Rev_Persistence",      # z-score direction × 28D magnitude
        ],
        # Theme 2: Historical surprise patterns (this company tends to beat/miss)
        "historical": [
            "Beat_Rate_4Q",         # rolling beat rate (persistence signal)
            "Consec_Beats",         # streak length (momentum)
            "Avg_Surprise_4Q",      # average surprise magnitude
            "Surprise_Trend_4Q",    # is surprise trajectory improving or degrading?
        ],
        # Theme 3: Earnings quality & fundamental trajectory
        "quality": [
            "Prev_EarningsQuality",     # ROE + margins - volatility composite
            "ROE_Momentum_Composite",   # ROE trend + revision agreement
            "Prev_ProfSurp_Composite",  # multi-metric surprise composite
            "Margin_Expansion",         # multiple margins improving together
        ],
        # Theme 4: Estimate gaps (consensus is stale relative to recent actuals)
        "estimate_gap": [
            "EPS_EstGap",           # prev actual vs current estimate
            "ROE_EstGap",           # ROE version
            "ROE_PrevAct_vs_Est",   # direct: last ROE actual minus this Q estimate
        ],
        # Theme 5: Industry early-reporter signal
        "industry": [
            "sep",                  # spread exceed probability (directional)
            "consensus_dir",        # industry beat direction
            "reporting_progress",   # how much of industry has reported
            "industry_certainty",   # confidence in industry signal
        ],
        # Theme 6: Cross-signal interactions (confirming signals amplify)
        "interaction": [
            "Rev_x_PrevSurp",          # revision × prev surprise (momentum confirmation)
            "RevSurp_Alignment",       # revision and prev surprise same direction?
            "Surprise_x_Streak",       # prev surprise × beat streak
            "Rev_x_ROE",               # revision × ROE level (quality confirmation)
            "SEP_x_PrevSurp",          # industry + company track record
        ],
        # Theme 7: Momentum & drift
        "momentum": [
            "C2O_Mom_2Q",          # post-earnings momentum (last 2 quarters)
            "C2O_Mom_4Q",          # post-earnings momentum (last 4 quarters)
            "RevZ_Mom_2Q",          # revision z-score momentum
        ],
        # Theme 8: Previous quarter specifics
        "prev_quarter": [
            "Prev_EPS_Surprise_Pct",   # last quarter's EPS surprise
            "Prev_Dual_Beat",          # did they beat on both EPS + revenue?
            "Prev_Triple_Beat",        # EPS + revenue + ROE
            "Prev_Surprise_Breadth",   # breadth of beats across metrics
        ],
    }

    # Filter to columns that actually exist in the dataframe
    available = set(df_columns)
    result = []
    for theme, feats in CURATED.items():
        for f in feats:
            if f in available:
                result.append(f)
    return result


def screen_features_by_ic(df, features, target, pcol, min_ic=0.02, min_hit_rate=0.55):
    """
    Pre-screen features: keep only those with |mean IC| > min_ic
    and directional consistency (hit rate) above threshold.

    This prevents the ranking model from fitting on noise features
    that happened to have high ICIR by chance.
    """
    kept = []
    for f in features:
        ics = []
        for p in df[pcol].unique():
            m = df[pcol] == p
            x, y = df.loc[m, f], df.loc[m, target]
            v = x.notna() & y.notna()
            if v.sum() < 10:
                continue
            r, _ = spearmanr(x[v], y[v])
            if not np.isnan(r):
                ics.append(r)
        if len(ics) < 4:
            continue
        mean_ic = np.mean(ics)
        hit = np.mean([1 if ic * np.sign(mean_ic) > 0 else 0 for ic in ics])
        if abs(mean_ic) >= min_ic and hit >= min_hit_rate:
            kept.append(f)
    return kept

# ═══════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white","axes.edgecolor":"#333",
    "axes.labelcolor":"#222","text.color":"#222","xtick.color":"#333","ytick.color":"#333",
    "grid.color":"#CCC","grid.alpha":.6,"grid.linestyle":"--",
    "legend.frameon":True,"legend.facecolor":"white","legend.edgecolor":"#CCC",
    "font.family":"serif","font.size":10,"axes.titlesize":12,"axes.labelsize":10,"figure.dpi":120,
})
POS,NEG,NEUT="#2E86AB","#D7263D","#888"
PAL=["#2E86AB","#D7263D","#F4A261","#2A9D8F","#7B2D8E","#E76F51","#264653","#E9C46A"]
QC=["#D7263D","#F4A261","#999","#2A9D8F","#2E86AB"]


# ═══════════════════════════════════════════════════════════════════
# CORE DIAGNOSTICS (same as v4, all guarded)
# ═══════════════════════════════════════════════════════════════════

def _corr(x, y, method="spearman"):
    v = x.notna() & y.notna()
    if v.sum() < 8: return np.nan
    xv, yv = x[v].values, y[v].values
    if xv.std() == 0 or yv.std() == 0: return np.nan
    return spearmanr(xv, yv)[0] if method == "spearman" else pearsonr(xv, yv)[0]

def compute_corr_panel(df, feats, target, pcol, method="spearman"):
    if len(df) == 0: return pd.DataFrame(columns=feats)
    recs = []
    for p in sorted(df[pcol].unique()):
        m = df[pcol] == p; y = df.loc[m, target]
        row = {"period": p}
        for f in feats: row[f] = _corr(df.loc[m, f], y, method)
        recs.append(row)
    if not recs: return pd.DataFrame(columns=feats)
    return pd.DataFrame(recs).set_index("period")

def compute_summary(panel, n_bh=None):
    _c = ["feature","mean","std","ir","t_stat","p_value","hit_rate","n_periods","bh_pvalue"]
    rows = []
    for f in panel.columns:
        ts = panel[f].dropna(); n = len(ts)
        if n < 3: continue
        m, s = ts.mean(), ts.std(ddof=1)
        ir = m/s if s > 1e-12 else 0; t = m/(s/np.sqrt(n)) if s > 1e-12 else 0
        p = 2*(1-t_dist.cdf(abs(t), df=n-1))
        hit = (ts*np.sign(m) > 0).mean() if abs(m) > 1e-12 else 0.5
        rows.append(dict(feature=f, mean=m, std=s, ir=ir, t_stat=t, p_value=p, hit_rate=hit, n_periods=n))
    if not rows: return pd.DataFrame(columns=_c)
    out = pd.DataFrame(rows).sort_values("ir", ascending=False, key=abs).reset_index(drop=True)
    pv = out["p_value"].values.copy(); nt = n_bh or len(pv)
    order = np.argsort(pv); bh = np.ones(len(pv))
    for i, idx in enumerate(order): bh[idx] = pv[idx]*nt/(i+1)
    si = np.argsort(pv); bs = bh[si]
    for i in range(len(bs)-2,-1,-1): bs[i] = min(bs[i], bs[i+1])
    bh[si] = bs; out["bh_pvalue"] = np.clip(bh, 0, 1)
    return out

def compute_quintile_returns(df, feats, target, pcol):
    results = {}
    for f in feats:
        recs = []
        for p in sorted(df[pcol].unique()):
            m = df[pcol]==p; x,y = df.loc[m,f], df.loc[m,target]; v = x.notna()&y.notna()
            if v.sum() < 10: continue
            try: q = pd.qcut(x[v], 5, labels=[1,2,3,4,5], duplicates="drop")
            except ValueError: continue
            for qi in range(1,6):
                qm = q==qi
                if qm.sum()>0: rv=y[v][qm]; recs.append(dict(period=p,quintile=qi,mean_ret=rv.mean(),hit_rate=(rv>0).mean(),n=qm.sum()))
        if recs: results[f] = pd.DataFrame(recs)
    return results

def compute_quintile_summary(qret):
    _c = ["feature","q1_ret","q2_ret","q3_ret","q4_ret","q5_ret","q1_hit","q2_hit","q3_hit","q4_hit","q5_hit","spread","monotonicity"]
    rows = []
    for f, qdf in qret.items():
        avg = qdf.groupby("quintile")["mean_ret"].mean(); hit = qdf.groupby("quintile")["hit_rate"].mean()
        vals = [avg.get(q,np.nan) for q in range(1,6)]; hits = [hit.get(q,np.nan) for q in range(1,6)]
        vv = [v for v in vals if np.isfinite(v)]
        mono = spearmanr(range(len(vv)),vv)[0] if len(vv)>=3 else np.nan
        rows.append(dict(feature=f,q1_ret=vals[0],q2_ret=vals[1],q3_ret=vals[2],q4_ret=vals[3],q5_ret=vals[4],
                         q1_hit=hits[0],q2_hit=hits[1],q3_hit=hits[2],q4_hit=hits[3],q5_hit=hits[4],
                         spread=avg.get(5,0)-avg.get(1,0),monotonicity=mono))
    if not rows: return pd.DataFrame(columns=_c)
    return pd.DataFrame(rows).sort_values("spread",ascending=False,key=abs).reset_index(drop=True)

def compute_conditional(df, feats, target, pcol, lo=0.25, hi=0.75):
    _c = ["feature","top_mean","top_hit","top_n","bot_mean","bot_hit","bot_n","spread","hit_spread"]
    rows = []
    for f in feats:
        top_r,bot_r=[],[]
        for p in sorted(df[pcol].unique()):
            m=df[pcol]==p; x,y=df.loc[m,f],df.loc[m,target]; v=x.notna()&y.notna()
            if v.sum()<8: continue
            xv,yv=x[v],y[v]; top_r.extend(yv[xv>=xv.quantile(hi)].tolist()); bot_r.extend(yv[xv<=xv.quantile(lo)].tolist())
        if len(top_r)<5 or len(bot_r)<5: continue
        ta,ba=np.array(top_r),np.array(bot_r)
        rows.append(dict(feature=f,top_mean=ta.mean(),top_hit=(ta>0).mean(),top_n=len(ta),
                         bot_mean=ba.mean(),bot_hit=(ba>0).mean(),bot_n=len(ba),
                         spread=ta.mean()-ba.mean(),hit_spread=(ta>0).mean()-(ba>0).mean()))
    if not rows: return pd.DataFrame(columns=_c)
    return pd.DataFrame(rows).sort_values("spread",ascending=False,key=abs).reset_index(drop=True)

def compute_walk_forward(df, feats, target, pcol, min_train=4):
    recs = []; periods = sorted(df[pcol].unique())
    for ti in range(min_train, len(periods)):
        m=df[pcol]==periods[ti]; y=df.loc[m,target]; row={"period":periods[ti]}
        for f in feats: row[f]=_corr(df.loc[m,f],y,"spearman")
        recs.append(row)
    if not recs: return pd.DataFrame(columns=feats)
    return pd.DataFrame(recs).set_index("period")

def compute_cumulative_dd(panel):
    if len(panel)==0 or len(panel.columns)==0:
        return pd.DataFrame(), pd.DataFrame(columns=["max_dd","dd_start","dd_end","dd_length"])
    cum = panel.cumsum(); rows = []
    for f in cum.columns:
        ts=cum[f].dropna()
        if len(ts)<2: rows.append(dict(feature=f,max_dd=0,dd_start=None,dd_end=None,dd_length=0)); continue
        pk=ts.expanding().max(); dd=ts-pk; mdd=dd.min()
        if mdd<0: de=dd.idxmin(); ds=ts.loc[:de].idxmax(); dl=list(ts.index).index(de)-list(ts.index).index(ds)
        else: ds=de=None; dl=0
        rows.append(dict(feature=f,max_dd=mdd,dd_start=ds,dd_end=de,dd_length=dl))
    return cum, pd.DataFrame(rows).set_index("feature")

def compute_pooled(df, feats, target, pcol, resid=True):
    _c=["feature","pooled_spearman","sp_p","pooled_pearson","pe_p","n"]
    if len(df)<20: return pd.DataFrame(columns=_c)
    d=df.copy(); tgt=target
    if resid:
        g=d.groupby(pcol)[target]
        if g.ngroups<1: return pd.DataFrame(columns=_c)
        d["_r"]=d[target]-g.transform("mean"); tgt="_r"
    rows=[]
    for f in feats:
        x,y=d[f],d[tgt]; v=x.notna()&y.notna()
        if v.sum()<20: continue
        rho,ps=spearmanr(x[v],y[v]); r,pp=pearsonr(x[v],y[v])
        rows.append(dict(feature=f,pooled_spearman=rho,sp_p=ps,pooled_pearson=r,pe_p=pp,n=v.sum()))
    if not rows: return pd.DataFrame(columns=_c)
    return pd.DataFrame(rows).sort_values("pooled_spearman",ascending=False,key=abs).reset_index(drop=True)

def compute_partial(df, feats, target, pcol):
    _c=["feature","standalone","partial","ctrl"]
    pooled=compute_pooled(df,feats,target,pcol)
    if len(pooled)<2: return pd.DataFrame(columns=_c)
    top_f=pooled.iloc[0]["feature"]; rows=[]
    for _,pr in pooled.iterrows():
        f=pr["feature"]
        if f==top_f: rows.append(dict(feature=f,standalone=pr["pooled_spearman"],partial=pr["pooled_spearman"],ctrl="—")); continue
        x=df[f].rank(pct=True); z=df[top_f].rank(pct=True); y=df[target]-df.groupby(pcol)[target].transform("mean")
        v=x.notna()&z.notna()&y.notna()
        if v.sum()<20: continue
        xv,zv,yv=x[v].values,z[v].values,y[v].values
        resid=xv-np.polyval(np.polyfit(zv,xv,1),zv)
        rows.append(dict(feature=f,standalone=pr["pooled_spearman"],partial=spearmanr(resid,yv)[0],ctrl=top_f))
    if not rows: return pd.DataFrame(columns=_c)
    return pd.DataFrame(rows).sort_values("partial",ascending=False,key=abs).reset_index(drop=True)

def compute_segments(df, feats, target, pcol, scol):
    if scol not in df.columns: return pd.DataFrame()
    parts=[]
    for seg in df[scol].dropna().unique():
        p=compute_pooled(df[df[scol]==seg],feats,target,pcol)
        if len(p)==0 or "pooled_spearman" not in p.columns: continue
        p["segment"]=seg; parts.append(p)
    if not parts: return pd.DataFrame()
    c=pd.concat(parts,ignore_index=True)
    if "pooled_spearman" not in c.columns: return pd.DataFrame()
    return c.pivot_table(index="feature",columns="segment",values="pooled_spearman").reindex(
        c.groupby("feature")["pooled_spearman"].apply(lambda x:x.abs().max()).sort_values(ascending=False).index)


# ═══════════════════════════════════════════════════════════════════
# CONFLUENCE VOTING SYSTEM
# ═══════════════════════════════════════════════════════════════════

class Voter:
    """
    A directional voter.  Returns (+1, confidence), (-1, confidence), or (0, 0).
    The func receives a row dict that includes pre-computed _pctl, _mean, _std columns.
    """
    def __init__(self, name, func, feature, category="custom"):
        self.name = name
        self.func = func
        self.feature = feature   # which column this voter reads (for preprocessing)
        self.category = category

    def vote(self, row):
        try:
            d, c = self.func(row)
            return int(np.sign(d)), float(np.clip(c, 0, 1))
        except:
            return 0, 0


def preprocess_voter_features(df, voters, pcol):
    """
    For each feature used by any voter, compute per-quarter:
      {feature}_pctl:  percentile rank within the quarter (0 to 1)
      {feature}_mean:  cross-sectional mean
      {feature}_std:   cross-sectional std

    These adaptive statistics replace hardcoded thresholds.
    A stock in the top 30% of revisions THIS quarter is treated the same
    regardless of whether this quarter's revisions are generally +1% or +10%.
    """
    df = df.copy()
    features_needed = set()
    for v in voters:
        if v.feature:
            features_needed.add(v.feature)

    for feat in features_needed:
        if feat not in df.columns:
            continue
        # Per-quarter percentile rank
        df[f"{feat}_pctl"] = df.groupby(pcol)[feat].rank(pct=True)
        # Per-quarter mean and std
        df[f"{feat}_mean"] = df.groupby(pcol)[feat].transform("mean")
        df[f"{feat}_std"] = df.groupby(pcol)[feat].transform("std").clip(lower=1e-8)
        # Z-score within quarter
        df[f"{feat}_z"] = (df[feat] - df[f"{feat}_mean"]) / df[f"{feat}_std"]

    return df


def build_default_voters():
    """
    5 ORTHOGONAL voters — each taps a genuinely independent information source.

    Design principles:
      - No two voters share the same underlying data (eliminates redundancy)
      - Each voter has a clear economic thesis for why it predicts direction
      - LONG bias by default (SHORT requires separate validation)
      - Higher dead zone (0.7 std) to reduce false signals
      - Confidence scales with extremity, maxes at 1.0

    Voter architecture:
      1. REVISION: Are analysts upgrading estimates? (Rev_CS_Full)
      2. PERSISTENCE: Does this company consistently beat? (Beat_Rate_4Q)
      3. INDUSTRY: Are early reporters in the industry beating? (sep)
      4. ESTIMATE GAP: Is consensus stale vs recent actuals? (EPS_EstGap)
      5. QUALITY: Is the earnings quality trajectory strong? (Prev_EarningsQuality)
    """
    voters = []

    # ── 1. REVISION MOMENTUM ──
    # Thesis: Analysts revising up → they've received positive channel checks,
    # company guided up, or sector peers beat. Cross-sectional rank avoids
    # comparing absolute revision magnitudes across different quarters.
    def _revision(row):
        z = row.get("Rev_CS_Full_z", 0)
        if abs(z) < 0.7: return 0, 0
        return np.sign(z), min(abs(z) / 2.5, 1)
    voters.append(Voter("revision_momentum", _revision, "Rev_CS_Full", "revision"))

    # ── 2. SURPRISE PERSISTENCE ──
    # Thesis: Companies that beat consistently have operational characteristics
    # (conservative guidance, execution quality) that persist quarter to quarter.
    # This is one of the best-documented signals in earnings literature.
    def _persistence(row):
        p = row.get("Beat_Rate_4Q_pctl", 0.5)
        if p > 0.75: return +1, min((p - 0.5) * 2, 1)
        elif p < 0.25: return -1, min((0.5 - p) * 2, 1)
        return 0, 0
    voters.append(Voter("surprise_persistence", _persistence, "Beat_Rate_4Q", "historical"))

    # ── 3. INDUSTRY EARLY-REPORTER SIGNAL ──
    # Thesis: Early reporters in an industry reveal common demand/pricing trends.
    # If banks reporting before JPM are beating, JPM probably will too.
    # Uses sep (spread exceed probability × direction).
    def _industry(row):
        z = row.get("sep_z", 0)
        if abs(z) < 0.7: return 0, 0
        return np.sign(z), min(abs(z) / 2.5, 1)
    voters.append(Voter("industry_signal", _industry, "sep", "industry"))

    # ── 4. ESTIMATE GAP ──
    # Thesis: When last quarter's actual EPS was much higher than this quarter's
    # consensus estimate, the street hasn't fully caught up. The gap represents
    # under-appreciated earnings power.
    def _estgap(row):
        z = row.get("EPS_EstGap_z", 0)
        if abs(z) < 0.7: return 0, 0
        return np.sign(z), min(abs(z) / 2.5, 1)
    voters.append(Voter("estimate_gap", _estgap, "EPS_EstGap", "estimate_gap"))

    # ── 5. EARNINGS QUALITY ──
    # Thesis: High ROE + improving margins + low volatility = sustainable
    # earnings power that produces positive surprises vs. estimates built
    # on lower-quality historical patterns.
    def _quality(row):
        p = row.get("Prev_EarningsQuality_pctl", 0.5)
        if p > 0.75: return +1, min((p - 0.5) * 2, 1)
        elif p < 0.25: return -1, min((0.5 - p) * 2, 1)
        return 0, 0
    voters.append(Voter("earnings_quality", _quality, "Prev_EarningsQuality", "quality"))

    return voters


def compute_confluence(df, voters, min_votes_to_trade=3, long_only=False):
    """
    For each row in df, run all voters, compute:
      - direction: net vote direction (+1 or -1), 0 if below threshold
      - conviction: sum of confidence-weighted votes / number of voting voters
      - n_long_votes, n_short_votes, n_abstain
      - top reasons (which voters drove the decision)

    Parameters:
      long_only: if True, only take LONG trades (skip SHORT even if qualified)

    Returns df with new columns added.
    """
    results = []
    for idx, row in df.iterrows():
        votes = []
        for v in voters:
            d, c = v.vote(row)
            votes.append({"voter": v.name, "category": v.category, "direction": d, "confidence": c})

        active = [v for v in votes if v["direction"] != 0]
        n_long = sum(1 for v in active if v["direction"] > 0)
        n_short = sum(1 for v in active if v["direction"] < 0)
        n_abstain = len(votes) - len(active)

        # Net direction
        long_conf = sum(v["confidence"] for v in active if v["direction"] > 0)
        short_conf = sum(v["confidence"] for v in active if v["direction"] < 0)
        net_conf = long_conf - short_conf
        net_votes = n_long - n_short

        # Decision
        if abs(net_votes) >= min_votes_to_trade and len(active) >= min_votes_to_trade:
            direction = 1 if net_votes > 0 else -1
            conviction = abs(net_conf) / max(len(active), 1)
            # Long-only filter: skip SHORT trades
            if long_only and direction == -1:
                direction = 0
                conviction = 0
        else:
            direction = 0  # no trade
            conviction = 0

        # Top reasons
        sorted_votes = sorted(active, key=lambda v: v["confidence"], reverse=True)
        reasons = "; ".join(f'{v["voter"]}({"+" if v["direction"]>0 else "-"}{v["confidence"]:.2f})'
                           for v in sorted_votes[:3])

        results.append(dict(
            _conf_direction=direction,
            _conf_conviction=round(conviction, 4),
            _conf_net_votes=net_votes,
            _conf_n_long=n_long,
            _conf_n_short=n_short,
            _conf_n_abstain=n_abstain,
            _conf_long_conf=round(long_conf, 4),
            _conf_short_conf=round(short_conf, 4),
            _conf_reasons=reasons,
        ))

    rdf = pd.DataFrame(results, index=df.index)
    return rdf


# ═══════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (shared by both approaches)
# ═══════════════════════════════════════════════════════════════════

def run_backtest_generic(df, direction_col, conviction_col, target, pcol,
                         date_col="announcement_date", initial_capital=1_000_000,
                         cost_bps=10, max_weight=0.15,
                         sizing="equal", label="strategy"):
    """
    Generic backtest given pre-computed direction and conviction columns.

    Sizing modes:
      "equal":      equal weight among all traded stocks on each date
      "conviction": weight proportional to conviction score
    """
    df = df.copy()
    # Only trade where direction != 0 AND target is not NaN
    df = df[(df[direction_col] != 0) & (df[target].notna())].copy()
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df.sort_values([pcol, date_col, "ticker"])
    cost_frac = cost_bps / 10000
    trade_dates = sorted(df[date_col].unique())

    all_trades = []
    capital = initial_capital

    for tdate in trade_dates:
        batch = df[df[date_col] == tdate]
        if len(batch) < 1: continue

        quarter = batch[pcol].iloc[0]
        dirs = batch[direction_col].values.astype(int)
        convs = batch[conviction_col].values.astype(float)
        rets = batch[target].values
        tickers = batch["ticker"].values
        reasons = batch["_conf_reasons"].values if "_conf_reasons" in batch.columns else [""] * len(batch)
        n = len(batch)
        n_long = (dirs > 0).sum()
        n_short = (dirs < 0).sum()

        # Compute weights
        if sizing == "conviction" and convs.sum() > 1e-12:
            weights = convs / convs.sum()
        else:
            weights = np.ones(n) / n

        # Cap per-name weight
        weights = np.minimum(weights, max_weight)
        # Do NOT renormalize to 100% — if few stocks, deploy less capital

        # PnL
        gross_rets = dirs * rets / 100
        net_rets = gross_rets - cost_frac
        pnls = capital * weights * net_rets
        total_pnl = pnls.sum()

        for j in range(n):
            all_trades.append(dict(
                strategy=label, date=tdate, quarter=quarter,
                ticker=tickers[j], direction=int(dirs[j]),
                conviction=round(convs[j], 4),
                weight=round(weights[j], 6),
                capital_pre=round(capital, 2),
                position_size=round(capital * weights[j], 2),
                return_pct=round(rets[j], 4),
                gross_pnl=round(capital * weights[j] * dirs[j] * rets[j] / 100, 2),
                cost=round(capital * weights[j] * cost_frac, 2),
                net_pnl=round(pnls[j], 2),
                hit=int(dirs[j] * rets[j] > 0),
                reasons=reasons[j] if j < len(reasons) else "",
            ))

        capital += total_pnl

    df_trades = pd.DataFrame(all_trades)
    if len(df_trades) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Aggregate
    df_daily = df_trades.groupby(["strategy","date","quarter"]).agg(
        n_trades=("net_pnl","count"),n_long=("direction",lambda x:(x>0).sum()),
        n_short=("direction",lambda x:(x<0).sum()),gross_pnl=("gross_pnl","sum"),
        total_cost=("cost","sum"),net_pnl=("net_pnl","sum"),hit_rate=("hit","mean"),
    ).reset_index()

    df_quarterly = df_trades.groupby(["strategy","quarter"]).agg(
        n_trades=("net_pnl","count"),n_long=("direction",lambda x:(x>0).sum()),
        n_short=("direction",lambda x:(x<0).sum()),gross_pnl=("gross_pnl","sum"),
        total_cost=("cost","sum"),net_pnl=("net_pnl","sum"),hit_rate=("hit","mean"),
        avg_conviction=("conviction","mean"),best_trade=("net_pnl","max"),worst_trade=("net_pnl","min"),
    ).reset_index()

    cum_cap = initial_capital; ret_pcts = []
    for _, row in df_quarterly.iterrows():
        ret_pcts.append(row["net_pnl"] / cum_cap * 100)
        cum_cap += row["net_pnl"]
    df_quarterly["return_pct"] = ret_pcts

    eq_recs = [dict(strategy=label, date="start", capital=initial_capital)]
    cap = initial_capital
    for _, row in df_daily.sort_values("date").iterrows():
        cap += row["net_pnl"]
        eq_recs.append(dict(strategy=label, date=row["date"], capital=cap))
    df_equity = pd.DataFrame(eq_recs)

    return df_trades, df_daily, df_quarterly, df_equity


# ═══════════════════════════════════════════════════════════════════
# TRADE STATISTICS — capital-independent signal evaluation
# ═══════════════════════════════════════════════════════════════════

def compute_trade_stats(df_trades, cost_bps=10):
    """
    Per-trade return statistics, independent of capital or compounding.

    For each trade:
        trade_return = direction × return_pct  (in %)
        trade_return_net = trade_return - cost_bps/100  (in %)

    Returns a dict of DataFrames:
        "per_trade":  every trade with trade_return columns added
        "overall":    single-row summary across all trades
        "by_direction": long vs short breakdown
        "by_quarter":  per-quarter stats (averaged, not compounded)
        "by_conviction": bucketed by conviction level
    """
    t = df_trades.copy()
    if len(t) == 0:
        return {"per_trade": pd.DataFrame(), "overall": pd.DataFrame(),
                "by_direction": pd.DataFrame(), "by_quarter": pd.DataFrame(),
                "by_conviction": pd.DataFrame()}

    cost_pct = cost_bps / 100  # e.g., 10bps = 0.10%

    # Core: per-trade return in % (capital-independent)
    t["trade_return"] = t["direction"] * t["return_pct"]
    t["trade_return_net"] = t["trade_return"] - cost_pct

    # ── Overall summary ──
    wins = t[t["trade_return_net"] > 0]
    losses = t[t["trade_return_net"] <= 0]
    overall = pd.DataFrame([dict(
        n_trades=len(t),
        avg_return=t["trade_return"].mean(),
        avg_return_net=t["trade_return_net"].mean(),
        hit_rate=(t["trade_return_net"] > 0).mean(),
        avg_win=wins["trade_return_net"].mean() if len(wins) > 0 else 0,
        avg_loss=losses["trade_return_net"].mean() if len(losses) > 0 else 0,
        win_loss_ratio=(abs(wins["trade_return_net"].mean() / losses["trade_return_net"].mean())
                        if len(wins) > 0 and len(losses) > 0 and losses["trade_return_net"].mean() != 0
                        else np.nan),
        profit_factor=(wins["trade_return_net"].sum() / abs(losses["trade_return_net"].sum())
                       if len(losses) > 0 and losses["trade_return_net"].sum() != 0
                       else np.nan),
        median_return=t["trade_return_net"].median(),
        std_return=t["trade_return_net"].std(),
        skew_return=t["trade_return_net"].skew(),
        best_trade=t["trade_return_net"].max(),
        worst_trade=t["trade_return_net"].min(),
        pct_long=(t["direction"] == 1).mean(),
    )])

    # ── By direction ──
    dir_rows = []
    for d, label in [(1, "LONG"), (-1, "SHORT")]:
        sub = t[t["direction"] == d]
        if len(sub) == 0: continue
        w = sub[sub["trade_return_net"] > 0]
        l = sub[sub["trade_return_net"] <= 0]
        dir_rows.append(dict(
            direction=label, n=len(sub),
            avg_return=sub["trade_return"].mean(),
            avg_return_net=sub["trade_return_net"].mean(),
            hit_rate=(sub["trade_return_net"] > 0).mean(),
            avg_win=w["trade_return_net"].mean() if len(w) > 0 else 0,
            avg_loss=l["trade_return_net"].mean() if len(l) > 0 else 0,
            profit_factor=(w["trade_return_net"].sum() / abs(l["trade_return_net"].sum())
                           if len(l) > 0 and l["trade_return_net"].sum() != 0 else np.nan),
        ))
    by_direction = pd.DataFrame(dir_rows)

    # ── By quarter (average trade return, NOT compounded) ──
    qg = t.groupby("quarter").agg(
        n_trades=("trade_return_net", "count"),
        avg_return=("trade_return", "mean"),
        avg_return_net=("trade_return_net", "mean"),
        hit_rate=("hit", "mean"),
        total_return=("trade_return_net", "sum"),
    ).reset_index()
    qg["avg_return_net_annualized"] = qg["avg_return_net"] * qg["n_trades"]  # sum per Q

    # ── By conviction bucket ──
    conv_rows = []
    if "conviction" in t.columns:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        t["_conv_bucket"] = pd.cut(t["conviction"], bins=bins, labels=labels, include_lowest=True)
        for bucket in labels:
            sub = t[t["_conv_bucket"] == bucket]
            if len(sub) < 3: continue
            conv_rows.append(dict(
                conviction_bucket=bucket, n=len(sub),
                avg_return_net=sub["trade_return_net"].mean(),
                hit_rate=(sub["trade_return_net"] > 0).mean(),
            ))
        t.drop(columns=["_conv_bucket"], inplace=True)
    by_conviction = pd.DataFrame(conv_rows)

    return {
        "per_trade": t,
        "overall": overall,
        "by_direction": by_direction,
        "by_quarter": qg,
        "by_conviction": by_conviction,
    }


def compute_date_returns(df_trades, date_col="date"):
    """
    Per-date portfolio return (weighted average of trade returns on each date).
    This is the proper time series for Sharpe computation.

    Returns DataFrame with one row per trade date:
        date, n_trades, portfolio_return (%), hit_rate
    """
    t = df_trades.copy()
    if len(t) == 0: return pd.DataFrame(columns=["date","n_trades","portfolio_return","hit_rate"])

    t["trade_return"] = t["direction"] * t["return_pct"]

    recs = []
    for d, grp in t.groupby(date_col):
        weights = grp["weight"].values
        tr = grp["trade_return"].values
        # Weighted portfolio return on this date
        wsum = weights.sum()
        if wsum > 1e-12:
            port_ret = (weights * tr).sum() / wsum
        else:
            port_ret = tr.mean()
        recs.append(dict(
            date=d, n_trades=len(grp),
            portfolio_return=port_ret,
            hit_rate=(grp["hit"]).mean(),
        ))
    return pd.DataFrame(recs)


def plot_trade_stats(stats, date_returns, label=""):
    """Plot per-trade statistics: return distribution, by-direction, by-conviction, date returns."""
    per_trade = stats["per_trade"]
    overall = stats["overall"]
    by_dir = stats["by_direction"]
    by_conv = stats["by_conviction"]
    if len(per_trade) == 0: return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) Trade return distribution
    ax = axes[0, 0]
    rets = per_trade["trade_return_net"].dropna().values
    ax.hist(rets, bins=30, color=POS, alpha=0.6, edgecolor="white")
    ax.axvline(0, c="black", lw=0.8)
    ax.axvline(np.mean(rets), c=NEG, lw=1.5, ls="--", label=f'mean={np.mean(rets):.3f}%')
    ax.axvline(np.median(rets), c=PAL[2], lw=1.5, ls=":", label=f'median={np.median(rets):.3f}%')
    ax.set_xlabel("Trade Return (%, net of costs)")
    ax.set_title(f"{label} Trade Return Distribution (n={len(rets)})")
    ax.legend(fontsize=8); ax.grid(axis="y")

    # (0,1) Long vs Short breakdown
    ax = axes[0, 1]
    if len(by_dir) > 0:
        x = range(len(by_dir))
        colors = [POS if d == "LONG" else NEG for d in by_dir["direction"]]
        ax.bar(x, by_dir["avg_return_net"], color=colors, alpha=0.7, width=0.5)
        ax.set_xticks(x)
        labels_list = []
        for _, row in by_dir.iterrows():
            labels_list.append(f'{row["direction"]}\nn={row["n"]}\nhit={row["hit_rate"]:.1%}\nPF={row["profit_factor"]:.2f}')
        ax.set_xticklabels(labels_list, fontsize=8)
        ax.axhline(0, c="black", lw=0.5)
        ax.set_ylabel("Avg Return (%, net)")
        ax.set_title("By Direction")
        ax.grid(axis="y")

    # (1,0) By conviction
    ax = axes[1, 0]
    if len(by_conv) > 0:
        x = range(len(by_conv))
        ax.bar(x, by_conv["avg_return_net"], color=POS, alpha=0.7, width=0.5)
        ax.set_xticks(x)
        lbl = [f'{r["conviction_bucket"]}\nn={r["n"]}\nhit={r["hit_rate"]:.1%}' for _, r in by_conv.iterrows()]
        ax.set_xticklabels(lbl, fontsize=8)
        ax.axhline(0, c="black", lw=0.5)
        ax.set_ylabel("Avg Return (%, net)")
        ax.set_title("By Conviction Level")
        ax.grid(axis="y")
    else:
        ax.text(0.5, 0.5, "No conviction data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("By Conviction Level")

    # (1,1) Date-level portfolio returns
    ax = axes[1, 1]
    if len(date_returns) > 0:
        dr = date_returns["portfolio_return"].values
        ax.bar(range(len(dr)), dr, color=[POS if v > 0 else NEG for v in dr], alpha=0.5, width=1)
        ax.axhline(0, c="black", lw=0.5)
        if len(dr) > 1:
            sharpe = np.mean(dr) / np.std(dr) * np.sqrt(min(len(dr), 250))
            ax.set_title(f"Per-Date Portfolio Return\nmean={np.mean(dr):.3f}% | Sharpe≈{sharpe:.2f}")
        else:
            ax.set_title("Per-Date Portfolio Return")
        ax.set_ylabel("Portfolio Return (%)")
        ax.set_xlabel(f"Trade Date (n={len(dr)})")
        ax.grid(axis="y")

    fig.suptitle(f"Trade Statistics: {label}", fontsize=12, y=1.01)
    _show(fig)


def plot_trade_stats_table(stats, date_returns, label=""):
    """Summary statistics as a table plot."""
    o = stats["overall"]
    if len(o) == 0: return

    dr = date_returns["portfolio_return"].values if len(date_returns) > 0 else np.array([0])
    n_dates = len(dr)
    date_sharpe = np.mean(dr) / np.std(dr) * np.sqrt(min(n_dates, 250)) if len(dr) > 1 and np.std(dr) > 0 else 0

    r = o.iloc[0]
    rows = [
        ["Total Trades", f'{r["n_trades"]:.0f}'],
        ["Avg Return (gross)", f'{r["avg_return"]:.3f}%'],
        ["Avg Return (net)", f'{r["avg_return_net"]:.3f}%'],
        ["Hit Rate", f'{r["hit_rate"]:.1%}'],
        ["Avg Win", f'{r["avg_win"]:.3f}%'],
        ["Avg Loss", f'{r["avg_loss"]:.3f}%'],
        ["Win/Loss Ratio", f'{r["win_loss_ratio"]:.2f}'],
        ["Profit Factor", f'{r["profit_factor"]:.2f}'],
        ["Median Return", f'{r["median_return"]:.3f}%'],
        ["Std Return", f'{r["std_return"]:.3f}%'],
        ["Best Trade", f'{r["best_trade"]:.2f}%'],
        ["Worst Trade", f'{r["worst_trade"]:.2f}%'],
        ["% Long", f'{r["pct_long"]:.1%}'],
        ["Trade Dates", f'{n_dates}'],
        ["Date Sharpe", f'{date_sharpe:.2f}'],
    ]

    fig, ax = plt.subplots(figsize=(5, len(rows) * 0.35 + 1))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   cellLoc="center", loc="center", colWidths=[0.5, 0.3])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#E0E0E0")
    ax.set_title(f"Signal Quality: {label}", fontsize=12, pad=15)
    _show(fig)

def _winsorize(arr, lo=0.05, hi=0.95):
    a = np.array(arr, dtype=float); v = ~np.isnan(a)
    if v.sum() < 5: return a
    return np.clip(a, np.nanpercentile(a, lo*100), np.nanpercentile(a, hi*100))

def compute_ranking_signal(df, feats, target, pcol, min_train=4, icir_threshold=0.3, max_features=15):
    """Walk-forward ICIR-weighted z-score signal."""
    periods = sorted(df[pcol].unique())
    signals = pd.Series(np.nan, index=df.index)
    wt_recs = []
    for ti in range(min_train, len(periods)):
        tp = periods[ti]; train = df[df[pcol].isin(periods[:ti])]
        panel = compute_corr_panel(train, feats, target, pcol, "spearman")
        weights = {}
        for f in feats:
            ts = panel[f].dropna()
            if len(ts) < 3: continue
            m, s = ts.mean(), ts.std(ddof=1)
            icir = m/s if s > 1e-12 else 0
            if abs(icir) > icir_threshold: weights[f] = icir
        if not weights: continue
        total = sum(abs(w) for w in weights.values())
        if total < 1e-12: continue
        weights = {f: w/total for f, w in weights.items()}
        wt_recs.append({"period": tp, **weights})
        mask = df[pcol] == tp; idx = df.index[mask]
        sig = np.zeros(mask.sum())
        for f, w in weights.items():
            vals = _winsorize(df.loc[mask, f].values.astype(float))
            mu, sd = np.nanmean(vals), np.nanstd(vals)
            z = (vals - mu) / max(sd, 1e-12); z = np.nan_to_num(z, 0)
            sig += w * z
        signals.loc[idx] = sig
    df_wt = pd.DataFrame(wt_recs)
    if len(df_wt) > 0: df_wt = df_wt.set_index("period")
    return signals, df_wt

def run_ranking_backtest(df, signal_col, target, pcol, date_col="announcement_date",
                         initial_capital=1_000_000, cost_bps=10, max_weight=0.15,
                         long_only=False, top_pct=0.20, bot_pct=0.20):
    """
    Ranking backtest using percentile thresholds per quarter.

    Parameters:
      long_only: if True, only go LONG on top-percentile stocks
      top_pct: percentile threshold for LONG (top 20% by default)
      bot_pct: percentile threshold for SHORT (bottom 20% by default)
    """
    df = df.copy(); df = df[df[signal_col].notna() & df[target].notna()].copy()
    df = df.sort_values([pcol, date_col, "ticker"])
    qt = {}
    for q in df[pcol].unique():
        qs = df.loc[df[pcol]==q, signal_col]
        if len(qs) < 4: continue
        qt[q] = {"top": qs.quantile(1 - top_pct), "bot": qs.quantile(bot_pct)}

    df["_rank_dir"] = 0
    df["_rank_conv"] = 0.0
    for q, th in qt.items():
        mask = df[pcol] == q
        df.loc[mask & (df[signal_col] >= th["top"]), "_rank_dir"] = 1
        if not long_only:
            df.loc[mask & (df[signal_col] <= th["bot"]), "_rank_dir"] = -1
        df.loc[mask, "_rank_conv"] = df.loc[mask, signal_col].abs()

    label = "ranking_long_only" if long_only else "ranking_long_short"
    df["_conf_reasons"] = ""
    return run_backtest_generic(df, "_rank_dir", "_rank_conv", target, pcol,
                                date_col, initial_capital, cost_bps, max_weight,
                                sizing="equal", label=label)


# ═══════════════════════════════════════════════════════════════════
# PLOTS (diagnostic — same as v4, all guarded)
# ═══════════════════════════════════════════════════════════════════

def _show(fig): fig.tight_layout(); plt.show(); plt.close(fig)

def plot_01_summary(s_sp, s_pe, top_n=25):
    if len(s_sp)==0: return
    d=s_sp.head(top_n); feats=d["feature"].values; pe=s_pe.set_index("feature")["mean"].to_dict() if len(s_pe)>0 else {}
    fig,ax=plt.subplots(figsize=(9,max(5,len(feats)*.3)))
    y=np.arange(len(feats)); sp=d["mean"].values
    ax.barh(y,sp,height=.6,color=[POS if v>0 else NEG for v in sp],alpha=.7,label="Spearman")
    ax.scatter([pe.get(f,0) for f in feats],y,c="black",s=18,zorder=5,marker="D",label="Pearson")
    ax.axvline(0,c="black",lw=.5)
    for i,row in d.iterrows():
        sig="***" if row["bh_pvalue"]<.01 else "**" if row["bh_pvalue"]<.05 else "*" if row["bh_pvalue"]<.10 else ""
        ax.annotate(f'IR={row["ir"]:.2f} {sig}',xy=(row["mean"],i),xytext=(4 if row["mean"]>=0 else -4,0),
                    textcoords="offset points",fontsize=6,ha="left" if row["mean"]>=0 else "right",va="center")
    ax.set_yticks(y); ax.set_yticklabels(feats,fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("Mean XS Correlation"); ax.set_title("Fig 1: Spearman vs Pearson\n*p<.10 **p<.05 ***p<.01",fontsize=10)
    ax.legend(fontsize=8,loc="lower right"); ax.grid(axis="x"); _show(fig)

def plot_02_timeseries(panel,s_sp,top_n=6):
    if len(s_sp)==0: return
    top=[f for f in s_sp.head(top_n)["feature"] if f in panel.columns]
    if not top: return
    fig,axes=plt.subplots(len(top),1,figsize=(11,2*len(top)),sharex=True)
    if len(top)==1: axes=[axes]
    for i,f in enumerate(top):
        ax=axes[i]; ts=panel[f].dropna()
        ax.bar(range(len(ts)),ts.values,color=[POS if v>0 else NEG for v in ts.values],alpha=.6,width=.8)
        if len(ts)>3: ax.plot(range(len(ts)),ts.rolling(4,min_periods=2).mean().values,c="black",lw=1.2,label="4Q roll")
        ax.axhline(0,c="black",lw=.4); m,s=ts.mean(),ts.std(); ir=m/s if s>1e-12 else 0
        ax.set_title(f"{f}  mean={m:.3f} IR={ir:.2f}",fontsize=9,loc="left"); ax.set_ylabel("ρ"); ax.legend(fontsize=7); ax.grid(axis="y")
        if i==len(top)-1: ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts.index,rotation=45,fontsize=6)
    fig.suptitle("Fig 2: Per-Period Spearman",fontsize=11,y=1.01); _show(fig)

def plot_03_quintiles(qret,s_sp,top_n=8):
    if len(s_sp)==0 or not qret: return
    top=[f for f in s_sp.head(top_n)["feature"] if f in qret][:top_n]
    if not top: return
    nc=min(4,len(top)); nr=max(1,(len(top)+nc-1)//nc)
    fig,axes=plt.subplots(nr,nc,figsize=(3.5*nc,3.2*nr)); axes=np.array(axes).flatten() if len(top)>1 else [axes]
    for i,f in enumerate(top):
        ax=axes[i]; avg=qret[f].groupby("quintile")["mean_ret"].mean(); hit=qret[f].groupby("quintile")["hit_rate"].mean()
        vals=[avg.get(q,0) for q in range(1,6)]; hits=[hit.get(q,.5) for q in range(1,6)]
        ax.bar(range(5),vals,color=QC,alpha=.75,width=.65)
        ax.set_xticks(range(5)); ax.set_xticklabels(["Q1","Q2","Q3","Q4","Q5"],fontsize=7)
        ax.axhline(0,c="black",lw=.4); ax.set_title(f"{f}\nsprd={vals[4]-vals[0]:.3f}",fontsize=8)
        ax.set_ylabel("Avg Δ%",fontsize=7); ax.grid(axis="y")
        for j,(v,h) in enumerate(zip(vals,hits)):
            ax.annotate(f'{h:.0%}',xy=(j,v),xytext=(0,3 if v>=0 else -9),textcoords="offset points",
                        fontsize=6,ha="center",color=POS if h>.55 else NEG if h<.45 else NEUT)
    for j in range(len(top),len(axes)): axes[j].set_visible(False)
    fig.suptitle("Fig 3: Quintile Analysis",fontsize=11); _show(fig)

def plot_04_conditional(cond,top_n=20):
    if len(cond)==0: return
    d=cond.head(top_n); feats=d["feature"].values
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,max(4,len(feats)*.28)))
    y=np.arange(len(feats)); w=.35
    a1.barh(y-w/2,d["top_mean"],w,color=POS,alpha=.7,label="Top Q"); a1.barh(y+w/2,d["bot_mean"],w,color=NEG,alpha=.7,label="Bot Q")
    a1.set_yticks(y); a1.set_yticklabels(feats,fontsize=7); a1.axvline(0,c="black",lw=.4)
    a1.set_xlabel("Mean Δ%"); a1.set_title("Conditional Mean"); a1.legend(fontsize=7); a1.invert_yaxis(); a1.grid(axis="x")
    a2.barh(y-w/2,d["top_hit"]*100,w,color=POS,alpha=.7,label="Top Q"); a2.barh(y+w/2,d["bot_hit"]*100,w,color=NEG,alpha=.7,label="Bot Q")
    a2.axvline(50,c="black",lw=.5,ls="--"); a2.set_yticks(y); a2.set_yticklabels(feats,fontsize=7)
    a2.set_xlabel("Hit Rate%"); a2.set_title("Conditional Hit Rate"); a2.legend(fontsize=7); a2.invert_yaxis(); a2.grid(axis="x")
    fig.suptitle("Fig 4: Top vs Bottom Quartile",fontsize=11); _show(fig)

def plot_05_walkforward(wf,wf_s,top_n=6):
    if len(wf_s)==0: return
    top=[f for f in wf_s.head(top_n)["feature"] if f in wf.columns]
    if not top: return
    fig,axes=plt.subplots(len(top),1,figsize=(11,1.8*len(top)),sharex=True)
    if len(top)==1: axes=[axes]
    for i,f in enumerate(top):
        ax=axes[i]; ts=wf[f].dropna()
        ax.bar(range(len(ts)),ts.values,color=[POS if v>0 else NEG for v in ts.values],alpha=.6,width=.8)
        ax.axhline(0,c="black",lw=.4); m,s=ts.mean(),ts.std(); ir=m/s if s>1e-12 else 0
        ax.set_title(f"[OOS] {f}  mean={m:.3f} IR={ir:.2f}",fontsize=9,loc="left"); ax.set_ylabel("ρ"); ax.grid(axis="y")
        if i==len(top)-1: ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts.index,rotation=45,fontsize=6)
    fig.suptitle("Fig 5: Walk-Forward OOS",fontsize=11,y=1.01); _show(fig)

def plot_06_cumulative(cum,dd,s_sp,top_n=8):
    if len(s_sp)==0 or len(cum)==0: return
    top=[f for f in s_sp.head(top_n)["feature"] if f in cum.columns]
    if not top: return
    fig,ax=plt.subplots(figsize=(11,5))
    for i,f in enumerate(top):
        ts=cum[f].dropna(); ddv=dd.loc[f,"max_dd"] if f in dd.index else 0
        ax.plot(range(len(ts)),ts.values,color=PAL[i%len(PAL)],lw=1.3,label=f'{f} (dd={ddv:.2f})')
    ax.axhline(0,c="black",lw=.4); ax.set_xticks(range(len(cum))); ax.set_xticklabels(cum.index,rotation=45,fontsize=6)
    ax.set_ylabel("Cum ρ"); ax.set_title("Fig 6: Cumulative Spearman"); ax.legend(fontsize=7,ncol=2); ax.grid(); _show(fig)

# ── Backtest plots ──

def plot_bt_equity(df_equity, initial_capital):
    if len(df_equity)==0: return
    fig,ax=plt.subplots(figsize=(12,5))
    for i,(strat,grp) in enumerate(df_equity.groupby("strategy")):
        grp=grp.sort_values("date"); ax.plot(range(len(grp)),grp["capital"].values,color=PAL[i%len(PAL)],lw=1.5,label=strat)
    ax.axhline(initial_capital,c="black",lw=.5,ls="--"); ax.set_ylabel("Capital ($)")
    ax.set_title("Equity Curve (compounding, after costs)"); ax.legend(fontsize=8); ax.grid(); _show(fig)

def plot_bt_quarterly(df_quarterly):
    if len(df_quarterly)==0: return
    strats=df_quarterly["strategy"].unique(); quarters=sorted(df_quarterly["quarter"].unique())
    fig,ax=plt.subplots(figsize=(13,5)); w=0.8/len(strats)
    for si,strat in enumerate(strats):
        sq=df_quarterly[df_quarterly["strategy"]==strat].set_index("quarter").reindex(quarters)
        off=(si-len(strats)/2+.5)*w; ax.bar(np.arange(len(quarters))+off,sq["net_pnl"].fillna(0).values,width=w,color=PAL[si%len(PAL)],alpha=.75,label=strat)
    ax.set_xticks(range(len(quarters))); ax.set_xticklabels(quarters,rotation=45,fontsize=7)
    ax.axhline(0,c="black",lw=.5); ax.set_ylabel("Net PnL ($)"); ax.set_title("Quarterly PnL")
    ax.legend(fontsize=7); ax.grid(axis="y"); _show(fig)

def plot_bt_summary(df_quarterly, df_equity, initial_capital):
    if len(df_quarterly)==0: return
    rows=[]
    for strat in df_quarterly["strategy"].unique():
        sq=df_quarterly[df_quarterly["strategy"]==strat]; eq=df_equity[df_equity["strategy"]==strat]
        final=eq["capital"].iloc[-1] if len(eq)>0 else initial_capital
        tr=(final/initial_capital-1)*100; nq=len(sq); ar=tr/max(nq/4,1)
        qr=sq["return_pct"].dropna(); sharpe=qr.mean()/qr.std()*2 if len(qr)>1 and qr.std()>0 else 0
        mdd=0
        if len(eq)>1:
            caps=eq["capital"].values; pk=np.maximum.accumulate(caps); mdd=((caps-pk)/pk*100).min()
        rows.append(dict(Strategy=strat,Final=f"${final:,.0f}",Return=f"{tr:.1f}%",Ann=f"{ar:.1f}%",
                         Sharpe=f"{sharpe:.2f}",MaxDD=f"{mdd:.1f}%",Hit=f"{sq['hit_rate'].mean()*100:.1f}%",
                         Trades=f"{sq['n_trades'].sum():,}",Costs=f"${sq['total_cost'].sum():,.0f}"))
    summary=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(12,1+len(rows)*.5)); ax.axis("off")
    tbl=ax.table(cellText=summary.values,colLabels=summary.columns,cellLoc="center",loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.5)
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor("#E0E0E0")
    ax.set_title("Backtest Summary",fontsize=12,pad=20); _show(fig)

def plot_confluence_votes(df_trades):
    """Show vote distribution and hit rate by conviction level."""
    if len(df_trades)==0: return
    t = df_trades.copy()
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))

    # Left: hit rate by net votes
    if "_conf_net_votes" in t.columns or "conviction" in t.columns:
        conv_bins = pd.cut(t["conviction"], bins=[0,.2,.4,.6,.8,1.01], labels=["0-.2",".2-.4",".4-.6",".6-.8",".8-1"])
        grp = t.groupby(conv_bins, observed=True).agg(hit=("hit","mean"),n=("hit","count")).reset_index()
        a1.bar(range(len(grp)),grp["hit"]*100,color=POS,alpha=.7)
        a1.set_xticks(range(len(grp))); a1.set_xticklabels(grp["conviction"],fontsize=8)
        for i,row in grp.iterrows():
            a1.annotate(f'n={row["n"]}',xy=(i,row["hit"]*100),xytext=(0,3),textcoords="offset points",fontsize=7,ha="center")
        a1.axhline(50,c="black",lw=.5,ls="--"); a1.set_ylabel("Hit Rate (%)")
        a1.set_xlabel("Conviction Bucket"); a1.set_title("Hit Rate by Conviction"); a1.grid(axis="y")

    # Right: PnL by direction
    dir_grp = t.groupby("direction").agg(net=("net_pnl","sum"),n=("net_pnl","count"),hit=("hit","mean")).reset_index()
    labels = ["SHORT" if d<0 else "LONG" for d in dir_grp["direction"]]
    colors = [NEG if d<0 else POS for d in dir_grp["direction"]]
    a2.bar(range(len(dir_grp)),dir_grp["net"],color=colors,alpha=.7)
    a2.set_xticks(range(len(dir_grp))); a2.set_xticklabels(labels)
    for i,row in dir_grp.iterrows():
        a2.annotate(f'n={row["n"]}, hit={row["hit"]:.0%}',xy=(i,row["net"]),xytext=(0,5),
                    textcoords="offset points",fontsize=8,ha="center")
    a2.axhline(0,c="black",lw=.5); a2.set_ylabel("Net PnL ($)"); a2.set_title("PnL by Direction"); a2.grid(axis="y")

    fig.suptitle("Confluence Vote Analysis",fontsize=12); _show(fig)


# ═══════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════

def _is_lookahead(col):
    """
    Returns True if a column uses CURRENT-QUARTER actuals (lookahead bias).

    Your dataset has two types of post-announcement data:
      1. Current-quarter actuals: ACT_*, *_Comp (without Prev_), *_raw margins
      2. Current-quarter derived: EPS_Surprise_Pct, EPS_Beat, Dual_Beat, etc.

    These are revealed WITH the earnings → cannot predict price_change.

    SAFE (pre-announcement):
      - EST_* (consensus estimates, known before)
      - Prev_* (previous quarter data, shifted by 1)
      - Avg_*_4Q (rolling averages of previous quarters)
      - Beat_Rate_4Q, ROE_Beat_Rate_4Q (rolling from shifted history)
      - Consec_Beats, Consec_Misses (from shifted beat flags)
      - REVISION_*, Rev_CS_Full, RevZ_* (analyst revision signals)
      - sep, beta_mean (industry SEP from prior reporters)
      - C2O_Mom_* (previous post-earnings momentum, shifted)
      - QoQ change features (shift(1) - shift(2), both previous)
      - Trend features (*_Trend_4Q, from shift(1).rolling)
      - ROE_Est_Level, ROE_PrevAct_vs_Est, ROE_Vol_4Q, etc.
      - All composite features built from the above
    """
    cl = col.strip()

    # ── 1. ACT_* columns: current-quarter reported actuals ──
    if cl.startswith("ACT_"):
        return True

    # ── 2. Price change / target variants ──
    if cl.lower() in ("price_change", "price_change_resid", "% price change"):
        return True

    # ── 3. Current-quarter *_Comp columns (actual composites) ──
    # ROE_Comp, Rev_Comp, GM_Comp, OpProf_Comp, CFPS_Comp, EPS_Adj_Comp, EPS_GAAP_Comp
    # But NOT Prev_* versions
    COMP_SUFFIXES = ("_Comp",)
    if any(cl.endswith(s) for s in COMP_SUFFIXES) and not cl.startswith("Prev_"):
        return True

    # ── 4. Current-quarter surprise columns ──
    # These are (ACT - EST) / EST computed for the CURRENT quarter.
    # Prev_* and Avg_*_4Q versions are safe (shifted/rolled from previous quarters).
    # Composite features containing "Surprise" or "Surp" in their name are also safe
    # (e.g., Surprise_x_Streak, Rev_x_PrevSurp, Prev_ProfSurp_Composite).
    CURRENT_SURPRISE_COLS = {
        "EPS_Surprise_Pct", "Rev_Surprise_Pct", "GM_Surprise_Pct",
        "ROE_Surprise_Pct", "ROE_Surprise_Raw",
        "CFPS_Surprise_Pct", "OpProf_Surprise_Pct",
        "EBITDA_Surprise_Pct", "EBIT_Surprise_Pct", "NI_Surprise_Pct",
        "BV_Surprise_Pct", "FCF_Surprise_Pct", "CapEx_Surprise_Pct",
        "TaxRate_Surprise", "SGA_Surprise_Pct", "COGS_Surprise_Pct",
    }
    if cl in CURRENT_SURPRISE_COLS:
        return True

    # ── 5. Current-quarter beat/miss flags ──
    # EPS_Beat, Rev_Beat, ROE_Beat, Dual_Beat, Triple_Beat
    # But NOT Prev_ROE_Beat, Prev_Dual_Beat, Prev_Triple_Beat (shifted = safe)
    # And NOT Beat_Rate_4Q, ROE_Beat_Rate_4Q (rolling from shifted = safe)
    # And NOT Consec_Beats, Consec_Misses (from shifted = safe)
    CURRENT_BEAT_COLS = {
        "EPS_Beat", "Rev_Beat", "ROE_Beat", "Dual_Beat", "Triple_Beat",
    }
    if cl in CURRENT_BEAT_COLS:
        return True

    # ── 6. Raw margin columns computed from current-quarter actuals ──
    # NetMargin_raw, EBITDA_Margin_raw, OpEx_Ratio_raw, EBIT_Margin_raw
    RAW_MARGIN_COLS = {
        "NetMargin_raw", "EBITDA_Margin_raw", "OpEx_Ratio_raw", "EBIT_Margin_raw",
    }
    if cl in RAW_MARGIN_COLS:
        return True

    # ── Everything else is pre-announcement ──
    return False

class Pipeline:
    def __init__(self, df, target="beta_adj_return", period_col="cal_quarter",
                 meta_cols=None, exclude_cols=None, segment_col="ann_type"):
        self.df = df.copy()
        self.target = target
        self.period_col = period_col
        self.segment_col = segment_col
        exclude = set(meta_cols or []); exclude.add(target)
        if exclude_cols: exclude.update(exclude_cols)
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        self.all_features = [c for c in numeric if c not in exclude]
        self.signal_features = [c for c in self.all_features if not _is_lookahead(c)]

        # Curated features for ranking (economic thesis, not kitchen-sink)
        self.curated_features = get_curated_features(df.columns)
        # Filter curated to only signal-safe (non-lookahead)
        self.curated_features = [f for f in self.curated_features if f in self.signal_features]

        # Result storage
        self.spearman_panel = self.pearson_panel = None
        self.summary_spearman = self.summary_pearson = None
        self.pooled = self.conditional = None
        self.quintile_returns_raw = self.quintile_summary = None
        self.walk_forward_panel = self.wf_summary = None
        self.cumulative = self.drawdowns = None
        self.partial = self.segments = None
        self.ranking_trades = self.ranking_quarterly = self.ranking_equity = None
        self.confluence_trades = self.confluence_quarterly = self.confluence_equity = None
        self.ranking_weights = None
        self.ranking_trade_stats = self.ranking_date_returns = None
        self.confluence_trade_stats = self.confluence_date_returns = None

    def run(self, min_train=4, plot=True, use_curated=True):
        """
        Run diagnostics.

        Parameters:
          use_curated: if True, run diagnostics on curated features only (recommended).
                       if False, run on all signal features (162+, noisy).
        """
        df = self.df; tgt = self.target; pcol = self.period_col
        feats = self.curated_features if use_curated else self.all_features
        nf = len(feats)
        print("="*64); print("  EARNINGS SIGNAL PIPELINE v6"); print("="*64)
        t = df[tgt]
        print(f"  {len(df)} obs | {df[pcol].nunique()} periods")
        print(f"  All features: {len(self.all_features)} | Signal: {len(self.signal_features)} | Curated: {len(self.curated_features)}")
        print(f"  Target: {tgt}  μ={t.mean():.4f}  σ={t.std():.4f}")
        print(f"  Using {'curated' if use_curated else 'all'} features for diagnostics ({nf})")
        if use_curated:
            print(f"  Curated features: {feats}")

        # ── Long vs Short baseline ──
        pos_rate = (t > 0).mean()
        print(f"\n  Baseline: P(positive return) = {pos_rate:.3f}")
        print(f"  ↳ Any strategy must beat {pos_rate:.1%} hit rate to add value on LONG side")

        print("\n[1] Spearman..."); self.spearman_panel = compute_corr_panel(df,feats,tgt,pcol,"spearman")
        print("[2] Pearson..."); self.pearson_panel = compute_corr_panel(df,feats,tgt,pcol,"pearson")
        print("[3] Summary..."); self.summary_spearman = compute_summary(self.spearman_panel,nf); self.summary_pearson = compute_summary(self.pearson_panel,nf)
        _pt("Spearman Summary",self.summary_spearman)
        print("[4] Pooled..."); self.pooled = compute_pooled(df,feats,tgt,pcol); _pt("Pooled",self.pooled,15)
        print("[5] Conditional..."); self.conditional = compute_conditional(df,feats,tgt,pcol); _pt("Conditional",self.conditional,15)
        print("[6] Quintiles..."); self.quintile_returns_raw = compute_quintile_returns(df,feats,tgt,pcol); self.quintile_summary = compute_quintile_summary(self.quintile_returns_raw)
        _pt("Quintile Spread", self.quintile_summary, 15)
        print(f"[7] Walk-forward..."); self.walk_forward_panel = compute_walk_forward(df,feats,tgt,pcol,min_train); self.wf_summary = compute_summary(self.walk_forward_panel,nf); _pt("Walk-Forward OOS",self.wf_summary,15)
        print("[8] Cumulative..."); self.cumulative, self.drawdowns = compute_cumulative_dd(self.spearman_panel)
        print("[9] Partial..."); self.partial = compute_partial(df,feats,tgt,pcol); _pt("Partial",self.partial,15)
        print("[10] Segments...")
        self.segments = pd.DataFrame()
        if self.segment_col and self.segment_col in df.columns:
            self.segments = compute_segments(df,feats,tgt,pcol,self.segment_col)

        # ── IC screening ──
        print("[11] IC screening (curated features)...")
        screened = screen_features_by_ic(df, self.curated_features, tgt, pcol,
                                          min_ic=0.02, min_hit_rate=0.55)
        print(f"  Passed IC screen: {len(screened)}/{len(self.curated_features)}")
        if screened:
            print(f"  Screened features: {screened}")
        self.screened_features = screened

        if plot:
            print("[12] Plotting...")
            plot_01_summary(self.summary_spearman,self.summary_pearson)
            plot_02_timeseries(self.spearman_panel,self.summary_spearman)
            plot_03_quintiles(self.quintile_returns_raw,self.summary_spearman)
            plot_04_conditional(self.conditional)
            plot_05_walkforward(self.walk_forward_panel,self.wf_summary)
            plot_06_cumulative(self.cumulative,self.drawdowns,self.summary_spearman)

        print("\n"+"="*64); print("  DIAGNOSTICS COMPLETE"); print("="*64)
        return self

    def backtest_ranking(self, capital=1_000_000, cost_bps=10, max_weight=0.15,
                         min_train=4, icir_threshold=0.3, max_features=10,
                         long_only=True, top_pct=0.20, plot=True):
        """
        Ranking backtest using curated + IC-screened features.

        Key changes from v5:
          - Uses curated features (not all 162)
          - Applies IC screening to reject noise features
          - long_only=True by default
          - top_pct=0.20 (top quintile, not quartile)
        """
        print("\n"+"="*64); print("  RANKING BACKTEST (v6)"); print("="*64)
        print(f"  Mode: {'LONG ONLY' if long_only else 'LONG/SHORT'}")
        print(f"  Top pct: {top_pct:.0%}")

        # Use screened features if available, else curated
        feats = self.screened_features if hasattr(self, 'screened_features') and self.screened_features else self.curated_features
        print(f"  Features for ranking: {len(feats)}")

        self.df["_rank_signal"], self.ranking_weights = compute_ranking_signal(
            self.df, feats, self.target, self.period_col,
            min_train, icir_threshold, max_features)
        n = self.df["_rank_signal"].notna().sum()
        print(f"  Signal for {n}/{len(self.df)} obs")

        if self.ranking_weights is not None and len(self.ranking_weights) > 0:
            # Show average feature weights
            avg_wts = self.ranking_weights.mean().abs().sort_values(ascending=False)
            print("  Avg feature weights (top 10):")
            for f, w in avg_wts.head(10).items():
                print(f"    {f:40s}  |w|={w:.4f}")

        self.ranking_trades, _, self.ranking_quarterly, self.ranking_equity = run_ranking_backtest(
            self.df, "_rank_signal", self.target, self.period_col, "announcement_date",
            capital, cost_bps, max_weight, long_only=long_only, top_pct=top_pct)
        _print_bt_summary(self.ranking_trades, self.ranking_quarterly,
                          self.ranking_equity, capital, "Ranking", cost_bps)

        self.ranking_trade_stats = {}
        self.ranking_date_returns = {}
        if len(self.ranking_trades) > 0:
            for strat in self.ranking_trades["strategy"].unique():
                st = self.ranking_trades[self.ranking_trades["strategy"] == strat]
                self.ranking_trade_stats[strat] = compute_trade_stats(st, cost_bps)
                self.ranking_date_returns[strat] = compute_date_returns(st)

        if plot and len(self.ranking_trades) > 0:
            plot_bt_equity(self.ranking_equity, capital)
            plot_bt_quarterly(self.ranking_quarterly)
            plot_bt_summary(self.ranking_quarterly, self.ranking_equity, capital)
            for strat in self.ranking_trade_stats:
                plot_trade_stats(self.ranking_trade_stats[strat],
                                self.ranking_date_returns[strat], strat)
                plot_trade_stats_table(self.ranking_trade_stats[strat],
                                      self.ranking_date_returns[strat], strat)
        return self

    def backtest_confluence(self, voters=None, min_votes=4, long_only=True,
                            capital=1_000_000, cost_bps=10, max_weight=0.15, plot=True):
        """
        Confluence directional backtest.

        Key changes from v5:
          - 5 orthogonal voters (not 8 redundant)
          - long_only=True by default
          - min_votes=4 (need 4 of 5 to agree, very selective)
        """
        if voters is None:
            voters = build_default_voters()

        print("\n"+"="*64); print("  CONFLUENCE BACKTEST (v6)"); print("="*64)
        print(f"  Mode: {'LONG ONLY' if long_only else 'LONG/SHORT'}")
        print(f"  Voters ({len(voters)}):")
        for v in voters: print(f"    {v.name:25s}  [{v.category:12s}]  → {v.feature}")
        print(f"  Min net votes to trade: {min_votes}")

        print("\n  Preprocessing: computing per-quarter percentiles, z-scores...")
        self.df = preprocess_voter_features(self.df, voters, self.period_col)
        print("  Computing votes...")
        conf_df = compute_confluence(self.df, voters, min_votes, long_only=long_only)
        self.df = pd.concat([self.df.drop(columns=[c for c in conf_df.columns if c in self.df.columns], errors="ignore"),
                             conf_df], axis=1)

        n_trade = (self.df["_conf_direction"] != 0).sum()
        n_long = (self.df["_conf_direction"] == 1).sum()
        n_short = (self.df["_conf_direction"] == -1).sum()
        n_skip = (self.df["_conf_direction"] == 0).sum()
        print(f"  Decisions: {n_long} LONG | {n_short} SHORT | {n_skip} NO TRADE")
        print(f"  Trade rate: {n_trade}/{len(self.df)} = {n_trade/len(self.df)*100:.1f}%")

        if n_trade > 0:
            print(f"  Average conviction (traded): {self.df.loc[self.df['_conf_direction']!=0, '_conf_conviction'].mean():.3f}")

        # ── Per-voter diagnostics ──
        print("\n  Per-voter hit rates (on traded subset):")
        traded_mask = self.df["_conf_direction"] != 0
        traded_df = self.df[traded_mask]
        for v in voters:
            feature_col = f"{v.feature}_z" if f"{v.feature}_z" in traded_df.columns else f"{v.feature}_pctl"
            if feature_col not in traded_df.columns:
                continue
            # Re-run voter on traded rows
            votes_arr = []
            for _, row in traded_df.iterrows():
                d, c = v.vote(row)
                votes_arr.append(d)
            votes_arr = np.array(votes_arr)
            target_vals = traded_df[self.target].values
            active = votes_arr != 0
            if active.sum() > 0:
                hit = (votes_arr[active] * target_vals[active] > 0).mean()
                print(f"    {v.name:25s}  voted={active.sum():4d}  hit={hit:.3f}")

        # Run backtests
        all_trades, all_quarterly, all_equity = [], [], []

        for sizing, label in [("equal", "conf_equal"), ("conviction", "conf_weighted")]:
            t, _, q, e = run_backtest_generic(
                self.df, "_conf_direction", "_conf_conviction", self.target,
                self.period_col, "announcement_date", capital, cost_bps, max_weight,
                sizing=sizing, label=label)
            if len(t) > 0:
                all_trades.append(t); all_quarterly.append(q); all_equity.append(e)

        if all_trades:
            self.confluence_trades = pd.concat(all_trades, ignore_index=True)
            self.confluence_quarterly = pd.concat(all_quarterly, ignore_index=True)
            self.confluence_equity = pd.concat(all_equity, ignore_index=True)
        else:
            self.confluence_trades = pd.DataFrame()
            self.confluence_quarterly = pd.DataFrame()
            self.confluence_equity = pd.DataFrame()

        _print_bt_summary(self.confluence_trades, self.confluence_quarterly,
                          self.confluence_equity, capital, "Confluence", cost_bps)

        self.confluence_trade_stats = {}
        self.confluence_date_returns = {}
        if len(self.confluence_trades) > 0:
            for strat in self.confluence_trades["strategy"].unique():
                st = self.confluence_trades[self.confluence_trades["strategy"] == strat]
                self.confluence_trade_stats[strat] = compute_trade_stats(st, cost_bps)
                self.confluence_date_returns[strat] = compute_date_returns(st)

        if plot and len(self.confluence_trades) > 0:
            plot_bt_equity(self.confluence_equity, capital)
            plot_bt_quarterly(self.confluence_quarterly)
            plot_bt_summary(self.confluence_quarterly, self.confluence_equity, capital)
            for strat in self.confluence_trade_stats:
                plot_trade_stats(self.confluence_trade_stats[strat],
                                self.confluence_date_returns[strat], strat)
                plot_trade_stats_table(self.confluence_trade_stats[strat],
                                      self.confluence_date_returns[strat], strat)
            if len(self.confluence_trades) > 0:
                plot_confluence_votes(self.confluence_trades)

        print("\n"+"="*64); print("  CONFLUENCE BACKTEST COMPLETE"); print("="*64)
        return self


def _print_bt_summary(trades, quarterly, equity, capital, label, cost_bps=10):
    if len(trades) == 0: print("  No trades."); return
    for strat in trades["strategy"].unique():
        st = trades[trades["strategy"]==strat]
        eq = equity[equity["strategy"]==strat]
        final = eq["capital"].iloc[-1] if len(eq) > 0 else capital
        tr = (final/capital-1)*100

        # Compute per-trade stats (capital-independent)
        stats = compute_trade_stats(st, cost_bps)
        o = stats["overall"].iloc[0] if len(stats["overall"]) > 0 else {}
        dr = compute_date_returns(st)

        print(f"\n  ── {strat} ──")
        print(f"    Compounding PnL: ${final:,.0f} ({tr:+.1f}%)")
        if len(o) > 0:
            print(f"    Per-Trade:  avg={o['avg_return_net']:.3f}%  hit={o['hit_rate']:.1%}"
                  f"  win/loss={o['win_loss_ratio']:.2f}  PF={o['profit_factor']:.2f}")
        bd = stats["by_direction"]
        if len(bd) > 0:
            for _, r in bd.iterrows():
                print(f"      {r['direction']:5s}: n={r['n']:<4.0f}  avg={r['avg_return_net']:.3f}%  hit={r['hit_rate']:.1%}"
                      f"  PF={r['profit_factor']:.2f}")
        if len(dr) > 1 and dr["portfolio_return"].std() > 0:
            d = dr["portfolio_return"]
            sharpe = d.mean() / d.std() * np.sqrt(min(len(d), 250))
            print(f"    Date-level: {len(dr)} dates  avg={d.mean():.3f}%  Sharpe={sharpe:.2f}")

def _pt(title, df, n=20):
    print(f"\n  ── {title} ──")
    if len(df)==0: print("    (no data)"); return
    with pd.option_context("display.float_format","{:.4f}".format,"display.max_rows",n,"display.width",140):
        print(df.head(n).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# DATA GENERATOR — matches real dataset schema
# ═══════════════════════════════════════════════════════════════════

def generate_sample(n_q=20, n_t=40, seed=42, na_rate=0.12):
    """
    Generate synthetic data matching the REAL earnings dataset schema.

    Columns produced mirror what the notebook feature-engineering pipeline
    outputs, so the Pipeline class, _is_lookahead(), and the default voters
    all work identically on both simulated and real data.

    Parameters
    ----------
    n_q : int     Number of quarters to simulate
    n_t : int     Number of tickers
    seed : int    Random seed
    na_rate : float  Approximate fraction of values to set NaN (realistic missingness)
    """
    rng = np.random.RandomState(seed)

    # ── Tickers & sector mapping ──
    tks = [
        ("AAPL","Apple","Tech","HW","Consumer Electronics"),
        ("MSFT","Microsoft","Tech","SW","Systems Software"),
        ("GOOGL","Alphabet","Tech","SW","Interactive Media"),
        ("AMZN","Amazon","CD","Retail","Internet Retail"),
        ("NVDA","NVIDIA","Tech","Semi","Semiconductors"),
        ("META","Meta","Tech","SW","Interactive Media"),
        ("TSLA","Tesla","CD","Auto","Automobile Manufacturers"),
        ("JPM","JPMorgan","Fin","Bank","Diversified Banks"),
        ("JNJ","J&J","HC","Pharma","Pharmaceuticals"),
        ("V","Visa","Fin","AM","Transaction Processing"),
        ("UNH","UnitedHealth","HC","MT","Managed Healthcare"),
        ("HD","Home Depot","CD","Retail","Home Improvement"),
        ("PG","P&G","CS","HPC","Household Products"),
        ("MA","Mastercard","Fin","AM","Transaction Processing"),
        ("LLY","Eli Lilly","HC","Pharma","Pharmaceuticals"),
        ("ABBV","AbbVie","HC","Bio","Biotechnology"),
        ("MRK","Merck","HC","MT","Pharmaceuticals"),
        ("AVGO","Broadcom","Tech","Semi","Semiconductors"),
        ("PEP","PepsiCo","CS","Food","Soft Drinks"),
        ("COST","Costco","CS","Retail","Hypermarkets"),
        ("TMO","Thermo Fisher","HC","MT","Life Sci Tools"),
        ("ADBE","Adobe","Tech","SW","Application Software"),
        ("CRM","Salesforce","Tech","SW","Application Software"),
        ("AMD","AMD","Tech","Semi","Semiconductors"),
        ("INTC","Intel","Tech","Semi","Semiconductors"),
        ("BA","Boeing","Ind","Aero","Aerospace & Defense"),
        ("CAT","Caterpillar","Ind","Mach","Farm & Heavy Equip"),
        ("GS","Goldman Sachs","Fin","Bank","Investment Banking"),
        ("MS","Morgan Stanley","Fin","Bank","Investment Banking"),
        ("BLK","BlackRock","Fin","AM","Asset Management"),
        ("ISRG","Intuitive Surg","HC","MT","Health Care Equipment"),
        ("RTX","RTX Corp","Ind","Aero","Aerospace & Defense"),
        ("DAL","Delta","Ind","Trans","Airlines"),
        ("UAL","United Airlines","Ind","Trans","Airlines"),
        ("WFC","Wells Fargo","Fin","Bank","Diversified Banks"),
        ("C","Citigroup","Fin","Bank","Diversified Banks"),
        ("PANW","Palo Alto","Tech","SW","Systems Software"),
        ("NOW","ServiceNow","Tech","SW","Systems Software"),
        ("UBER","Uber","Tech","SW","Application Software"),
        ("AXP","AmEx","Fin","Bank","Consumer Finance"),
    ][:n_t]

    sector_beta = {"Tech": .3, "HC": .1, "Fin": -.05, "CD": .05, "CS": .02, "Ind": 0}
    id_days = {"Bank":(1,6),"AM":(1,8),"Pharma":(5,12),"Semi":(8,16),"SW":(10,18),
               "HW":(10,18),"MT":(8,15),"Bio":(8,15),"Retail":(15,25),"Auto":(12,20),
               "Aero":(10,18),"Mach":(10,18),"Trans":(12,20),"HPC":(10,18),"Food":(10,18)}

    # ── ACT / EST metric definitions ──
    ACT_EST_METRICS = {
        "9_EARNINGS_PER_SHARE":        ("eps",    (0.5, 5),     0.12, 0.04),
        "20_SALES":                    ("rev",    (1000, 50000),0.08, 0.03),
        "27_GROSS_MARGIN_GROSS_PROFIT_MARGIN": ("gm", (20, 65), 0.05, 0.02),
        "6_EARNINGS_BEFORE_INTEREST_AND_TAXES": ("ebit", (50, 5000), 0.10, 0.04),
        "8_EARNINGS_BEFORE_INTEREST_TAXES_AND_DEPRECIATION": ("ebitda", (80, 6000), 0.10, 0.04),
        "15_NET_INCOME":               ("ni",     (20, 4000),   0.12, 0.04),
        "1_BOOK_VALUE_PER_SHARE":      ("bv",     (5, 80),      0.06, 0.02),
        "2_CASH_FLOW_PER_SHARE":       ("cfps",   (1, 15),      0.10, 0.04),
        "104_OPERATING_EXPENSE":       ("opex",   (200, 15000), 0.08, 0.03),
        "109_INTEREST_EXPENSE":        ("intexp", (5, 500),     0.10, 0.05),
        "12_AS_REPORTED_EPS":          ("areps",  (0.4, 5),     0.12, 0.04),
        "13_NET_ASSETS_VALUE":         ("nav",    (10, 100),    0.08, 0.03),
        "14_NET_DEBT":                 ("netdbt", (100, 30000), 0.10, 0.05),
        "153_SHAREHOLDERS_EQUITY":     ("sheq",   (500, 50000), 0.06, 0.03),
        "157_TOTAL_ASSETS":            ("tassets",(2000,100000), 0.05, 0.02),
        "17_PRETAX_PROFIT":            ("ptp",    (30, 5000),   0.10, 0.04),
        "219_SG_A_EXPENSE":            ("sga",    (50, 8000),   0.08, 0.03),
        "22_CAPITAL_EXPENDITURES":     ("capex",  (20, 5000),   0.12, 0.05),
        "230_COST_OF_GOODS_SOLD":      ("cogs",   (300, 30000), 0.06, 0.03),
        "232_DEPRECIATION_AMORTIZATION":("da",    (20, 3000),   0.08, 0.03),
        "237_FREE_CASH_FLOW":          ("fcf",    (10, 5000),   0.15, 0.06),
        "238_GROSS_INCOME":            ("gi",     (200, 20000), 0.08, 0.03),
        "245_NUMBER_OF_SHARES_OUTSTANDING":("shares",(100,5000),0.01, 0.005),
        "24_REPORTED_NET_PROFIT":      ("rnp",    (20, 4000),   0.12, 0.04),
        "258_TAX_PROVISION":           ("taxprov",(5, 1500),    0.12, 0.05),
        "259_TAX_RATE":                ("taxrate",(10, 30),     0.05, 0.02),
        "28_REPORTED_PRETAX_PROFIT":   ("rptp",   (30, 5000),   0.10, 0.04),
        "4_DIVIDEND_PER_SHARE":        ("dps",    (0, 3),       0.10, 0.03),
    }
    ACT_ONLY = {
        "227_CASH_FLOW_FROM_FINANCING":("cffi",  (-5000, 500),  0.15),
        "228_CASH_FLOW_FROM_INVESTING":("cfinv", (-8000, -100), 0.12),
        "229_CASH_FLOW_FROM_OPERATIONS":("cfop", (50, 8000),    0.10),
        "235_EBITDA_REPORTED":         ("ebitdar",(80, 6000),   0.10),
        "239_GOODWILL":                ("gw",    (0, 30000),    0.05),
        "257_TOTAL_DIVIDENDS":         ("divtot",(0, 3000),     0.08),
        "351_CASH_MKTBLSEC":           ("cash",  (500, 40000),  0.08),
        "406_CURRENT_ASSETS":          ("ca",    (1000, 60000), 0.06),
        "407_CURRENT_LIABILITIES":     ("cl",    (500, 30000),  0.06),
    }
    EST_ONLY = {
        "19_RETURN_ON_EQUITY": ("roe_est_raw", (5, 40), 0.03),
        "351_CASH_MKTBLSEC":  ("cash_est", (500, 40000), 0.04),
        "406_CURRENT_ASSETS":  ("ca_est", (1000, 60000), 0.03),
        "407_CURRENT_LIABILITIES": ("cl_est", (500, 30000), 0.03),
    }

    # ── Build quarters ──
    qs = [(2021 + q // 4, q % 4 + 1) for q in range(n_q)]
    tb = {t[0]: .4 + rng.rand() * .5 for t in tks}
    ts = {t[0]: rng.randn() * 5 for t in tks}
    bh = {t[0]: [] for t in tks}
    rows = []

    def _maybe_nan(val, rate=na_rate):
        return np.nan if rng.rand() < rate else val

    for yr, qn in qs:
        ql = f"{yr}Q{qn}"
        bm = (qn - 1) * 3 + 4; by = yr if bm <= 12 else yr + 1; bm = bm if bm <= 12 else bm - 12
        sp = rng.randn(30) * .005
        itr = {}
        order = list(range(len(tks))); rng.shuffle(order)
        order.sort(key=lambda i: id_days.get(tks[i][3], (10, 18))[0] + rng.randint(0, 3))

        for idx in order:
            tk, nm, sec, ind, sub_ind = tks[idx]
            dlo, dhi = id_days.get(ind, (10, 18))
            ao = rng.randint(dlo, dhi + 1); ad = min(max(ao, 1), 28)
            sr = float(sp[min(ao, 29)])

            row = dict(
                ticker=tk, name=nm, sector=sec, industry=ind, sub_industry=sub_ind,
                announcement_date=f"{by}-{bm:02d}-{ad:02d}",
                ann_type=rng.choice(["AMC", "BMO"]),
                cal_quarter=ql,
            )

            # ── ACT / EST pairs ──
            for key, (tag, (lo, hi), na, ne) in ACT_EST_METRICS.items():
                base = lo + rng.rand() * (hi - lo)
                act = base * (1 + rng.randn() * na)
                est = base * (1 + rng.randn() * ne)
                row[f"ACT_{key}"] = _maybe_nan(round(act, 4), na_rate * 0.3)
                row[f"EST_{key}"] = _maybe_nan(round(est, 4), na_rate * 0.3)

            # ── ACT-only ──
            for key, (tag, (lo, hi), na_) in ACT_ONLY.items():
                base = lo + rng.rand() * (hi - lo)
                row[f"ACT_{key}"] = _maybe_nan(round(base * (1 + rng.randn() * na_), 4), na_rate * 0.4)

            # ── EST-only ──
            for key, (tag, (lo, hi), ne_) in EST_ONLY.items():
                base = lo + rng.rand() * (hi - lo)
                est_key = f"EST_{key}"
                if est_key not in row:
                    row[est_key] = _maybe_nan(round(base * (1 + rng.randn() * ne_), 4), na_rate * 0.4)

            # ── Comp / Est pairs ──
            eps_act = row.get("ACT_9_EARNINGS_PER_SHARE", 2.0) or 2.0
            eps_est = row.get("EST_9_EARNINGS_PER_SHARE", 2.0) or 2.0
            rev_act = row.get("ACT_20_SALES", 10000) or 10000
            rev_est = row.get("EST_20_SALES", 10000) or 10000
            gm_act  = row.get("ACT_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN", 40) or 40
            gm_est  = row.get("EST_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN", 40) or 40

            row["EPS_Adj_Comp"] = _maybe_nan(round(eps_act * (1 + rng.randn() * .01), 4))
            row["EPS_Adj_Est"]  = _maybe_nan(round(eps_est * (1 + rng.randn() * .005), 4))
            row["EPS_GAAP_Comp"]= _maybe_nan(round(eps_act * (1 + rng.randn() * .02), 4))
            row["EPS_GAAP_Est"] = _maybe_nan(round(eps_est * (1 + rng.randn() * .01), 4))
            row["Rev_Comp"]     = round(rev_act, 4)
            row["Rev_Est"]      = round(rev_est, 4)
            row["GM_Comp"]      = round(gm_act, 4)
            row["GM_Est"]       = round(gm_est, 4)

            roe_base = 5 + rng.rand() * 35
            row["ROE_Comp"]     = _maybe_nan(round(roe_base * (1 + rng.randn() * .08), 4))
            row["ROE_Est"]      = _maybe_nan(round(roe_base * (1 + rng.randn() * .03), 4))
            opprof = row.get("ACT_6_EARNINGS_BEFORE_INTEREST_AND_TAXES", 500) or 500
            row["OpProf_Comp"]  = _maybe_nan(round(opprof * (1 + rng.randn() * .05), 4))
            row["OpProf_Est"]   = _maybe_nan(round(opprof * (1 + rng.randn() * .02), 4))
            cfps = row.get("ACT_2_CASH_FLOW_PER_SHARE", 5) or 5
            row["CFPS_Comp"]    = _maybe_nan(round(cfps * (1 + rng.randn() * .08), 4))
            row["CFPS_Est"]     = _maybe_nan(round(cfps * (1 + rng.randn() * .03), 4))

            # ── Revision columns ──
            rd_ = rng.randn() * 3
            for w in [3, 7, 14, 28, 56]:
                row[f"REVISION_{w}D"]      = _maybe_nan(round(rd_ + rng.randn() * 1.5, 4))
                row[f"REVISION_PCT_{w}D"]  = _maybe_nan(round((rd_ + rng.randn() * 1.5) * 0.5, 4))
            row["REVISION_ZSCORE"]   = _maybe_nan(round(rd_ / max(abs(rng.randn() * 2 + 1), 0.5), 4))
            row["REVISION_ACCEL"]    = _maybe_nan(round((row.get("REVISION_PCT_3D", 0) or 0) -
                                                        (row.get("REVISION_PCT_14D", 0) or 0), 4))
            row["REVISION_VOL_13W"]  = _maybe_nan(round(abs(rng.randn()) * 0.05 + 0.01, 4))

            # ── SEP / industry ──
            ik = f"{ind}_{ql}"
            if ik not in itr:
                itr[ik] = dict(tot=sum(1 for t in tks if t[3] == ind), pos=0, neg=0, rep=0)
            tr_ = itr[ik]
            ap, bp = tr_["pos"] + 1, tr_["neg"] + 1
            bmv = ap / (ap + bp)
            sep_ = round(1 - beta_dist.cdf(.5 / 1.5, ap, bp), 4)
            bvar = round(abs(rng.randn() * 0.1), 4)

            row["n_industry_total"] = tr_["tot"]
            row["n_reported"]       = tr_["rep"]
            row["sep"]              = _maybe_nan(round(sep_ * (2 * bmv - 1), 4))
            row["beta_mean"]        = round(bmv, 4)
            row["beta_var"]         = bvar

            # ── SP500 return ──
            row["sp500_ret"] = round(sr * 100, 4)

            # ── Target: price_change ──
            eb = rng.rand() < tb[tk]; bh[tk].append(float(eb))
            sig = (sector_beta.get(sec, 0)
                   + (.3 if (row.get("REVISION_ZSCORE") or 0) > 0 else -.2) * .5
                   + (.2 if bmv > .5 else -.15) * .3 * sep_
                   + (.5 if eb else -.4) * 1.2
                   + .3 * sr * 100
                   + rng.randn() * 2.2)
            row["price_change"] = round(sig, 4)

            if row["price_change"] > 0: tr_["pos"] += 1
            else: tr_["neg"] += 1
            tr_["rep"] += 1
            rows.append(row)

    df = pd.DataFrame(rows)

    # ═══════════════════════════════════════════════════════════
    # DERIVED FEATURES (replicating the notebook pipeline)
    # ═══════════════════════════════════════════════════════════
    df["price_change_resid"] = df["price_change"] - df["sp500_ret"]

    # ── Surprise metrics ──
    def _surp(act, est):
        return ((act - est) / est.abs().replace(0, np.nan) * 100).clip(-200, 200)

    df["EPS_Surprise_Pct"] = _surp(df["ACT_9_EARNINGS_PER_SHARE"], df["EST_9_EARNINGS_PER_SHARE"])
    df["EPS_Beat"] = (df["EPS_Surprise_Pct"] > 0).astype(float).where(df["EPS_Surprise_Pct"].notna(), np.nan)
    df["Rev_Surprise_Pct"] = _surp(df["Rev_Comp"], df["Rev_Est"])
    rev_beat = (df["Rev_Surprise_Pct"] > 0).astype(float).where(df["Rev_Surprise_Pct"].notna(), np.nan)
    df["Dual_Beat"] = ((df["EPS_Beat"] == 1) & (rev_beat == 1)).astype(float).where(
        df["EPS_Beat"].notna() & rev_beat.notna(), np.nan)
    df["GM_Surprise_Pct"] = (df["ACT_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN"] -
                              df["EST_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN"]) * 100

    df["ROE_Surprise_Pct"] = _surp(df["ROE_Comp"], df["ROE_Est"])
    df["ROE_Surprise_Raw"] = df["ROE_Comp"] - df["ROE_Est"]
    roe_beat = (df["ROE_Surprise_Pct"] > 0).astype(float).where(df["ROE_Surprise_Pct"].notna(), np.nan)

    for act_k, est_k, name in [
        ("CFPS_Comp", "CFPS_Est", "CFPS_Surprise_Pct"),
        ("OpProf_Comp", "OpProf_Est", "OpProf_Surprise_Pct"),
        ("ACT_8_EARNINGS_BEFORE_INTEREST_TAXES_AND_DEPRECIATION",
         "EST_8_EARNINGS_BEFORE_INTEREST_TAXES_AND_DEPRECIATION", "EBITDA_Surprise_Pct"),
        ("ACT_6_EARNINGS_BEFORE_INTEREST_AND_TAXES",
         "EST_6_EARNINGS_BEFORE_INTEREST_AND_TAXES", "EBIT_Surprise_Pct"),
        ("ACT_15_NET_INCOME", "EST_15_NET_INCOME", "NI_Surprise_Pct"),
        ("ACT_1_BOOK_VALUE_PER_SHARE", "EST_1_BOOK_VALUE_PER_SHARE", "BV_Surprise_Pct"),
        ("ACT_237_FREE_CASH_FLOW", "EST_237_FREE_CASH_FLOW", "FCF_Surprise_Pct"),
        ("ACT_22_CAPITAL_EXPENDITURES", "EST_22_CAPITAL_EXPENDITURES", "CapEx_Surprise_Pct"),
        ("ACT_219_SG_A_EXPENSE", "EST_219_SG_A_EXPENSE", "SGA_Surprise_Pct"),
        ("ACT_230_COST_OF_GOODS_SOLD", "EST_230_COST_OF_GOODS_SOLD", "COGS_Surprise_Pct"),
    ]:
        if act_k in df.columns and est_k in df.columns:
            df[name] = _surp(df[act_k], df[est_k])

    df["TaxRate_Surprise"] = df["ACT_259_TAX_RATE"] - df["EST_259_TAX_RATE"]

    # ── Raw margins ──
    safe_rev = df["Rev_Comp"].abs().replace(0, np.nan)
    df["NetMargin_raw"]     = np.where(safe_rev.notna(), df["ACT_15_NET_INCOME"] / df["Rev_Comp"] * 100, np.nan)
    df["EBITDA_Margin_raw"] = np.where(safe_rev.notna(), df["ACT_8_EARNINGS_BEFORE_INTEREST_TAXES_AND_DEPRECIATION"] / df["Rev_Comp"] * 100, np.nan)
    df["OpEx_Ratio_raw"]    = np.where(safe_rev.notna(), df["ACT_104_OPERATING_EXPENSE"] / df["Rev_Comp"] * 100, np.nan)
    df["EBIT_Margin_raw"]   = np.where(safe_rev.notna(), df["ACT_6_EARNINGS_BEFORE_INTEREST_AND_TAXES"] / df["Rev_Comp"] * 100, np.nan)

    # ── Sort by ticker + quarter for rolling calcs ──
    df = df.sort_values(["ticker", "cal_quarter"]).reset_index(drop=True)
    g = df.groupby("ticker")

    # Beat & surprise rolling
    df["Avg_Surprise_4Q"] = g["EPS_Surprise_Pct"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    df["Prev_Quarter_Surprise"] = g["EPS_Surprise_Pct"].transform(lambda s: s.shift(1))
    df["Beat_Rate_4Q"] = g["EPS_Beat"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())

    # Previous-Q values
    for col, src in [
        ("Prev_ROE", "ROE_Comp"), ("Prev_EPS", "ACT_9_EARNINGS_PER_SHARE"),
        ("Prev_Gross_Margin", "ACT_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN"),
        ("Prev_Net_Margin", "NetMargin_raw"), ("Prev_EBITDA_Margin", "EBITDA_Margin_raw"),
        ("Prev_Book_Value", "ACT_1_BOOK_VALUE_PER_SHARE"),
        ("Prev_OpEx_Ratio", "OpEx_Ratio_raw"), ("Prev_Tax_Rate", "ACT_259_TAX_RATE"),
        ("Prev_EBIT_Margin", "EBIT_Margin_raw"), ("Prev_CFPS", "ACT_2_CASH_FLOW_PER_SHARE"),
    ]:
        if src in df.columns:
            df[col] = g[src].shift(1)

    # QoQ changes
    df["ROE_Change_QoQ"] = g["ROE_Comp"].shift(1) - g["ROE_Comp"].shift(2)
    df["GM_Change_QoQ"]  = g["ACT_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN"].shift(1) - g["ACT_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN"].shift(2)
    df["NetMargin_Change_QoQ"] = g["NetMargin_raw"].shift(1) - g["NetMargin_raw"].shift(2)
    df["BV_Growth_QoQ"] = ((g["ACT_1_BOOK_VALUE_PER_SHARE"].shift(1) /
                             g["ACT_1_BOOK_VALUE_PER_SHARE"].shift(2).replace(0, np.nan)) - 1) * 100
    eps_s = df["ACT_9_EARNINGS_PER_SHARE"]
    df["EPS_Growth_QoQ"] = ((g[eps_s.name].shift(1) - g[eps_s.name].shift(2)) /
                             g[eps_s.name].shift(2).abs().replace(0, np.nan) * 100).clip(-500, 500)
    df["EPS_Growth_YoY"] = ((g[eps_s.name].shift(1) - g[eps_s.name].shift(5)) /
                             g[eps_s.name].shift(5).abs().replace(0, np.nan) * 100).clip(-500, 500)

    # C2O momentum
    df["C2O_Mom_2Q"] = g["price_change"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).sum())
    df["C2O_Mom_4Q"] = g["price_change"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).sum())

    # Consecutive beats / misses
    def _consec(s, target_val):
        shifted = s.shift(1)
        result = pd.Series(np.nan, index=s.index)
        streak = 0
        for i in range(len(shifted)):
            v = shifted.iloc[i]
            if pd.isna(v): streak = 0; result.iloc[i] = np.nan
            elif v == target_val: streak += 1; result.iloc[i] = float(streak)
            else: streak = 0; result.iloc[i] = 0.0
        return result
    df["Consec_Beats"]  = g["EPS_Beat"].transform(lambda s: _consec(s, 1.0))
    df["Consec_Misses"] = g["EPS_Beat"].transform(lambda s: _consec(s, 0.0))

    # ── SEP derived ──
    df["consensus_dir"] = np.where(df["beta_mean"].isna(), np.nan,
                                   np.where(df["beta_mean"] > 0.5, 1.0, -1.0))
    df["reporting_progress"] = (df["n_reported"] / df["n_industry_total"].clip(lower=1)).clip(0, 1)
    df["industry_certainty"] = (1.0 / (1.0 + df["beta_var"] * df["n_industry_total"].clip(lower=1))).clip(0, 1)

    # ── Revision composites ──
    def _cs_rank(col):
        return df.groupby("cal_quarter")[col].rank(pct=True) - 0.5

    for base, cs_name in [("REVISION_ZSCORE", "RevZ_CS"), ("REVISION_PCT_3D", "RevPct3_CS"),
                           ("REVISION_PCT_7D", "RevPct7_CS"), ("REVISION_PCT_14D", "RevPct14_CS"),
                           ("REVISION_ACCEL", "RevAccel_CS")]:
        if base in df.columns:
            df[cs_name] = _cs_rank(base)

    df["Rev_CS_Full"] = (df.get("RevZ_CS", 0) * 0.40 + df.get("RevPct3_CS", 0) * 0.25 +
                          df.get("RevPct7_CS", 0) * 0.20 + df.get("RevAccel_CS", 0) * 0.15)
    if "REVISION_ZSCORE" in df.columns and "REVISION_VOL_13W" in df.columns:
        df["RevZ_VolAdj"] = (df["REVISION_ZSCORE"] / df["REVISION_VOL_13W"].clip(lower=0.001, upper=0.10)).clip(-20, 20)
    df["Rev_Accel_3v14"] = df.get("REVISION_PCT_3D", 0) - df.get("REVISION_PCT_14D", 0)
    df["Rev_Sign_Consensus"] = (np.sign(df.get("REVISION_PCT_3D", pd.Series(0, index=df.index)).fillna(0)) +
                                 np.sign(df.get("REVISION_PCT_7D", pd.Series(0, index=df.index)).fillna(0)) +
                                 np.sign(df.get("REVISION_PCT_14D", pd.Series(0, index=df.index)).fillna(0))) / 3.0
    df["Rev_Persistence"] = (np.sign(df.get("REVISION_ZSCORE", pd.Series(0, index=df.index)).fillna(0)) *
                              df.get("REVISION_PCT_28D", pd.Series(0, index=df.index)).abs()).clip(-0.30, 0.30)

    # Revision momentum
    df["RevZ_Mom_2Q"] = g["REVISION_ZSCORE"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    df["RevZ_Mom_4Q"] = g["REVISION_ZSCORE"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())

    def _slope(s):
        v = s.dropna()
        return float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) >= 3 else np.nan
    df["Rev_Trend_4Q"] = g["REVISION_ZSCORE"].transform(lambda s: s.shift(1).rolling(4, min_periods=3).apply(_slope, raw=False))
    df["RevPct3_Trend_4Q"] = g["REVISION_PCT_3D"].transform(lambda s: s.shift(1).rolling(4, min_periods=3).apply(_slope, raw=False))

    df["Surprise_Trend_4Q"] = g["EPS_Surprise_Pct"].transform(
        lambda s: s.shift(2).rolling(4, min_periods=3).apply(
            lambda v: float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) >= 3 else np.nan, raw=False))

    # ── Previous-quarter surprises (shifted by 1) ──
    surp_cols = ["EPS_Surprise_Pct", "Rev_Surprise_Pct", "GM_Surprise_Pct",
                 "ROE_Surprise_Pct", "CFPS_Surprise_Pct", "OpProf_Surprise_Pct",
                 "EBITDA_Surprise_Pct", "EBIT_Surprise_Pct", "NI_Surprise_Pct",
                 "BV_Surprise_Pct", "FCF_Surprise_Pct", "TaxRate_Surprise",
                 "SGA_Surprise_Pct", "COGS_Surprise_Pct", "CapEx_Surprise_Pct"]
    for col_name in surp_cols:
        if col_name in df.columns:
            df[f"Prev_{col_name}"] = g[col_name].shift(1)

    for col_name in ["Rev_Surprise_Pct", "ROE_Surprise_Pct", "GM_Surprise_Pct",
                     "CFPS_Surprise_Pct", "EBITDA_Surprise_Pct"]:
        if col_name in df.columns:
            df[f"Avg_{col_name}_4Q"] = g[col_name].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    df["Avg_EPS_Surprise_Pct_4Q"] = df["Avg_Surprise_4Q"]

    # ── Beat flags ──
    df["ROE_Beat"] = roe_beat
    df["Rev_Beat"] = rev_beat
    df["Prev_ROE_Beat"] = g["ROE_Beat"].shift(1)
    df["Prev_Rev_Beat"] = g["Rev_Beat"].shift(1)
    df["Prev_Dual_Beat"] = g["Dual_Beat"].shift(1)
    df["ROE_Beat_Rate_4Q"] = g["ROE_Beat"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    df["Triple_Beat"] = ((df["EPS_Beat"] == 1) & (rev_beat == 1) & (roe_beat == 1)).astype(float)
    df["Triple_Beat"] = df["Triple_Beat"].where(df["EPS_Beat"].notna() & rev_beat.notna() & roe_beat.notna(), np.nan)
    df["Prev_Triple_Beat"] = g["Triple_Beat"].shift(1)

    # ── ROE composites ──
    df["ROE_Est_Level"] = df["ROE_Est"]
    df["ROE_PrevAct_vs_Est"] = g["ROE_Comp"].shift(1) - df["ROE_Est"]
    df["ROE_Trend_4Q"] = g["ROE_Comp"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=3).apply(_slope, raw=False))
    df["ROE_Vol_4Q"] = g["ROE_Comp"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())
    df["ROE_CV_4Q"] = df["ROE_Vol_4Q"] / g["ROE_Comp"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean()).abs().replace(0, np.nan)
    df["ROE_MeanRev"] = g["ROE_Comp"].transform(
        lambda s: s.shift(1) - s.shift(1).rolling(4, min_periods=2).mean())
    df["Prev_ROE_EPS_DualBeat"] = g["ROE_Beat"].shift(1) * g["EPS_Beat"].shift(2)

    # ── Margin trends ──
    for src, trg in [("NetMargin_raw", "NetMargin_Trend_4Q"),
                     ("EBITDA_Margin_raw", "EBITDA_Margin_Trend_4Q"),
                     ("EBIT_Margin_raw", "EBIT_Margin_Trend_4Q"),
                     ("OpEx_Ratio_raw", "OpEx_Ratio_Trend_4Q")]:
        if src in df.columns:
            df[trg] = g[src].transform(lambda s: s.shift(1).rolling(4, min_periods=3).apply(_slope, raw=False))

    # ── Surprise breadth ──
    _beat_cols_temp = []
    for bname, acol, ecol in [("_b_ebitda", "ACT_8_EARNINGS_BEFORE_INTEREST_TAXES_AND_DEPRECIATION",
                                "EST_8_EARNINGS_BEFORE_INTEREST_TAXES_AND_DEPRECIATION"),
                               ("_b_ebit", "ACT_6_EARNINGS_BEFORE_INTEREST_AND_TAXES",
                                "EST_6_EARNINGS_BEFORE_INTEREST_AND_TAXES"),
                               ("_b_ni", "ACT_15_NET_INCOME", "EST_15_NET_INCOME"),
                               ("_b_fcf", "ACT_237_FREE_CASH_FLOW", "EST_237_FREE_CASH_FLOW")]:
        if acol in df.columns and ecol in df.columns:
            s = _surp(df[acol], df[ecol])
            df[bname] = (s > 0).astype(float).where(s.notna(), np.nan)
            _beat_cols_temp.append(bname)

    _prev_beat_df = g[_beat_cols_temp].shift(1) if _beat_cols_temp else pd.DataFrame(index=df.index)
    df["Prev_Surprise_Breadth"] = _prev_beat_df.mean(axis=1) if len(_beat_cols_temp) > 0 else np.nan
    df.drop(columns=_beat_cols_temp, inplace=True, errors="ignore")

    # ── Interaction / composite features ──
    df["Rev_x_PrevSurp"] = df["REVISION_ZSCORE"] * df["Prev_EPS_Surprise_Pct"].fillna(0) / 100
    df["Rev_x_PrevROESurp"] = df["REVISION_ZSCORE"] * df.get("Prev_ROE_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) / 100
    df["Rev_ROE_Weighted"] = df["REVISION_ZSCORE"] * (1 / (1 + df["ROE_Vol_4Q"].fillna(1)))

    df["Prev_EarningsQuality"] = (
        df["Prev_ROE"].fillna(0) / 30 +
        df["Prev_Gross_Margin"].fillna(0) / 100 +
        df["Prev_Net_Margin"].fillna(0) / 30 -
        df["ROE_Vol_4Q"].fillna(0.5) / 5
    ).clip(-3, 3)

    df["Prev_ProfSurp_Composite"] = (
        df.get("Prev_ROE_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) / 100 * 0.4 +
        df.get("Prev_GM_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) / 100 * 0.3 +
        df.get("Prev_EBITDA_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) / 100 * 0.3
    ).clip(-2, 2)

    df["ROE_Momentum_Composite"] = (
        df["ROE_Trend_4Q"].fillna(0) / 2 +
        df["REVISION_ZSCORE"].fillna(0) / 3
    ).clip(-3, 3)

    df["Margin_Expansion"] = (
        df["NetMargin_Change_QoQ"].fillna(0) +
        df["ROE_Change_QoQ"].fillna(0)
    ).clip(-30, 30)

    # Estimate gaps
    df["EPS_EstGap"] = ((g["ACT_9_EARNINGS_PER_SHARE"].shift(1) - df["EST_9_EARNINGS_PER_SHARE"]) /
                         df["EST_9_EARNINGS_PER_SHARE"].abs().replace(0, np.nan) * 100).clip(-200, 200)
    df["ROE_EstGap"] = ((g["ROE_Comp"].shift(1) - df["ROE_Est"]) /
                         df["ROE_Est"].abs().replace(0, np.nan) * 100).clip(-200, 200)
    df["GM_EstGap"] = (g["ACT_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN"].shift(1) -
                        df["EST_27_GROSS_MARGIN_GROSS_PROFIT_MARGIN"])

    df["Rev_x_ROE"] = df["REVISION_ZSCORE"] * df["Prev_ROE"].fillna(0) / 30
    df["Surprise_x_Streak"] = df.get("Prev_EPS_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) / 100 * df["Consec_Beats"].fillna(0)
    df["Prev_OpLeverage"] = (df.get("Prev_EBIT_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) -
                              df.get("Prev_Rev_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0)).clip(-100, 100)
    df["Prev_CostDiscipline"] = -(df.get("Prev_SGA_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) +
                                   df.get("Prev_COGS_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0)) / 200
    df["Prev_CashConversion"] = (df.get("Prev_FCF_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) -
                                  df.get("Prev_NI_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0)).clip(-100, 100)

    df["RevSurp_Alignment"] = np.sign(df["REVISION_ZSCORE"].fillna(0)) * np.sign(df.get("Prev_EPS_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0))
    df["RevSurp_ROE_Alignment"] = np.sign(df["REVISION_ZSCORE"].fillna(0)) * np.sign(df.get("Prev_ROE_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0))
    df["ROE_ExpectationRisk"] = df["ROE_Est"].fillna(0) / 30 * df["ROE_Vol_4Q"].fillna(0.5)

    df["MultiMetric_AvgSurp_4Q"] = (
        df["Avg_EPS_Surprise_Pct_4Q"].fillna(0) / 100 * 0.4 +
        df.get("Avg_Rev_Surprise_Pct_4Q", pd.Series(0, index=df.index)).fillna(0) / 100 * 0.3
    ).clip(-2, 2)

    df["SEP_x_PrevSurp"] = df["sep"].fillna(0) * df.get("Prev_EPS_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) / 100
    df["SEP_x_ROETrend"] = df["sep"].fillna(0) * df["ROE_Trend_4Q"].fillna(0)
    df["Rev_x_MarginTrend"] = df["REVISION_ZSCORE"] * (
        df["GM_Change_QoQ"].fillna(0) + df["NetMargin_Change_QoQ"].fillna(0)) / 10

    df["C2O_x_PrevSurp"] = df["C2O_Mom_2Q"].fillna(0) * df.get("Prev_EPS_Surprise_Pct", pd.Series(0, index=df.index)).fillna(0) / 100

    df["ROE_EstDispersion"] = (df.get("EST_19_RETURN_ON_EQUITY", pd.Series(np.nan, index=df.index)) - df["ROE_Est"]).abs()
    df["ROE_vs_Sector"] = df["Prev_ROE"] - df.groupby(["cal_quarter", "sector"])["Prev_ROE"].transform("median")
    df["ROE_Accel"] = df["ROE_Change_QoQ"] - g["ROE_Change_QoQ"].shift(1)

    # ── Revision Regime (categorical) ──
    if "REVISION_ZSCORE" in df.columns:
        df["Revision_Regime"] = pd.cut(df["REVISION_ZSCORE"],
            bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
            labels=["Strong Down", "Down", "Flat", "Up", "Strong Up"])

    return df


# ═══════════════════════════════════════════════════════════════════
# MAIN — v6 usage pattern
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    REAL DATA USAGE:
    ─────────────────────────────────────────────────────────────
    # 1. Load your pre-processed DataFrame
    df = pd.read_excel('earnings_df.xlsx')
    # (assumes all feature engineering from notebook is already applied)

    # 2. Beta-adjust the target (replaces naive price_change - sp500_ret)
    df = compute_beta_adjusted_target(df)

    # 3. Define metadata columns to exclude
    meta = ["ticker", "name", "sector", "industry", "sub_industry",
            "announcement_date", "ann_type", "cal_quarter", "ann_date",
            "Unnamed: 0", "Year/Period", "Announcement Time",
            "TICKER", "PERENDDATE", "ANNOUNCEDATE", "CONSENSUSSNAPDATE",
            "Ann Type", "Revision_Regime"]
    exclude = ["sp500_ret", "price_change", "price_change_resid", "_ticker_beta"]

    # 4. Run pipeline with curated features
    pipe = Pipeline(df, target="beta_adj_return", period_col="cal_quarter",
                    meta_cols=meta, exclude_cols=exclude)
    pipe.run(use_curated=True)

    # 5. Ranking backtest — long only, top quintile
    pipe.backtest_ranking(capital=1_000_000, long_only=True, top_pct=0.20)

    # 6. Confluence backtest — long only, 4 of 5 voters must agree
    pipe.backtest_confluence(min_votes=4, long_only=True, capital=1_000_000)

    # 7. If long-only shows edge, test adding shorts cautiously:
    # pipe.backtest_confluence(min_votes=4, long_only=False, capital=1_000_000)
    ─────────────────────────────────────────────────────────────
    """

    # For testing with simulated data (produces real-schema columns):
    df = generate_sample(n_q=20, seed=42)

    # Beta-adjust the target
    df = compute_beta_adjusted_target(df)

    meta = ["ticker", "name", "sector", "industry", "sub_industry",
            "announcement_date", "ann_type", "cal_quarter", "Revision_Regime"]
    exclude = ["sp500_ret", "price_change", "price_change_resid", "_ticker_beta"]

    pipe = Pipeline(df, target="beta_adj_return", period_col="cal_quarter",
                    meta_cols=meta, exclude_cols=exclude)
    pipe.run(use_curated=True)
    pipe.backtest_ranking(capital=1_000_000, long_only=True, top_pct=0.20)
    pipe.backtest_confluence(min_votes=4, long_only=True, capital=1_000_000)