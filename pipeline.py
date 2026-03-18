"""
Earnings Signal Pipeline
========================

USAGE:
    import pandas as pd
    from pipeline import Pipeline

    df = pd.read_csv("earnings_dataset.csv")

    # Optional: add composite features — just new columns
    df["eps_norm"] = df["est_eps"] / df.groupby("industry")["est_eps"].transform("std")

    # Define which columns are metadata (NOT features)
    meta = ["ticker", "name", "sector", "industry", "subindustry",
            "ann_date", "ann_type", "cal_quarter"]

    pipe = Pipeline(df, target="price_change", period_col="cal_quarter", meta_cols=meta)
    pipe.run()

    # All results stored as DataFrames — inspect after run:
    pipe.summary_spearman
    pipe.pooled
    pipe.conditional
    pipe.quintile_summary
    pipe.walk_forward_panel
    pipe.wf_summary
    pipe.partial
    pipe.segments
    pipe.drawdowns
    pipe.spearman_panel
    pipe.pearson_panel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, t as t_dist
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# STYLE — clean academic
# ═══════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333", "axes.labelcolor": "#222", "text.color": "#222",
    "xtick.color": "#333", "ytick.color": "#333",
    "grid.color": "#CCC", "grid.alpha": 0.6, "grid.linestyle": "--",
    "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "#CCC",
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10, "figure.dpi": 120,
})

POS, NEG, NEUT = "#2E86AB", "#D7263D", "#888"
PAL = ["#2E86AB", "#D7263D", "#F4A261", "#2A9D8F", "#7B2D8E", "#E76F51", "#264653", "#E9C46A"]
QC = ["#D7263D", "#F4A261", "#999", "#2A9D8F", "#2E86AB"]


# ═══════════════════════════════════════════════════════════════════
# COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════

def _corr(x, y, method="spearman"):
    valid = x.notna() & y.notna()
    if valid.sum() < 8: return np.nan
    xv, yv = x[valid].values, y[valid].values
    if xv.std() == 0 or yv.std() == 0: return np.nan
    return spearmanr(xv, yv)[0] if method == "spearman" else pearsonr(xv, yv)[0]


def compute_corr_panel(df, feats, target, pcol, method="spearman"):
    periods = sorted(df[pcol].unique())
    recs = []
    for p in periods:
        m = df[pcol] == p
        y = df.loc[m, target]
        row = {"period": p}
        for f in feats:
            row[f] = _corr(df.loc[m, f], y, method)
        recs.append(row)
    return pd.DataFrame(recs).set_index("period")


def compute_summary(panel, n_bh=None):
    rows = []
    for f in panel.columns:
        ts = panel[f].dropna()
        n = len(ts)
        if n < 3: continue
        m, s = ts.mean(), ts.std(ddof=1)
        ir = m / s if s > 1e-12 else 0
        t = m / (s / np.sqrt(n)) if s > 1e-12 else 0
        p = 2 * (1 - t_dist.cdf(abs(t), df=n - 1))
        hit = (ts * np.sign(m) > 0).mean() if abs(m) > 1e-12 else 0.5
        rows.append(dict(feature=f, mean=m, std=s, ir=ir, t_stat=t,
                         p_value=p, hit_rate=hit, n_periods=n))
    out = pd.DataFrame(rows).sort_values("ir", ascending=False, key=abs).reset_index(drop=True)
    if len(out) > 0:
        pv = out["p_value"].values.copy()
        nt = n_bh or len(pv)
        order = np.argsort(pv)
        bh = np.ones(len(pv))
        for i, idx in enumerate(order): bh[idx] = pv[idx] * nt / (i + 1)
        s_idx = np.argsort(pv); bh_s = bh[s_idx]
        for i in range(len(bh_s) - 2, -1, -1): bh_s[i] = min(bh_s[i], bh_s[i+1])
        bh[s_idx] = bh_s
        out["bh_pvalue"] = np.clip(bh, 0, 1)
    return out


def compute_quintile_returns(df, feats, target, pcol):
    results = {}
    for f in feats:
        recs = []
        for p in sorted(df[pcol].unique()):
            m = df[pcol] == p
            x, y = df.loc[m, f], df.loc[m, target]
            v = x.notna() & y.notna()
            if v.sum() < 10: continue
            try: q = pd.qcut(x[v], 5, labels=[1,2,3,4,5], duplicates="drop")
            except ValueError: continue
            for qi in range(1, 6):
                qm = q == qi
                if qm.sum() > 0:
                    rv = y[v][qm]
                    recs.append(dict(period=p, quintile=qi, mean_ret=rv.mean(),
                                     hit_rate=(rv > 0).mean(), n=qm.sum()))
        if recs: results[f] = pd.DataFrame(recs)
    return results


def compute_quintile_summary(qret):
    rows = []
    for f, qdf in qret.items():
        avg = qdf.groupby("quintile")["mean_ret"].mean()
        hit = qdf.groupby("quintile")["hit_rate"].mean()
        vals = [avg.get(q, np.nan) for q in range(1, 6)]
        hits = [hit.get(q, np.nan) for q in range(1, 6)]
        spread = (avg.get(5, 0) - avg.get(1, 0))
        vv = [v for v in vals if np.isfinite(v)]
        mono = spearmanr(range(len(vv)), vv)[0] if len(vv) >= 3 else np.nan
        rows.append(dict(feature=f, q1_ret=vals[0], q2_ret=vals[1], q3_ret=vals[2],
                         q4_ret=vals[3], q5_ret=vals[4], q1_hit=hits[0], q2_hit=hits[1],
                         q3_hit=hits[2], q4_hit=hits[3], q5_hit=hits[4],
                         spread=spread, monotonicity=mono))
    return pd.DataFrame(rows).sort_values("spread", ascending=False, key=abs).reset_index(drop=True)


def compute_conditional(df, feats, target, pcol, lo=0.25, hi=0.75):
    rows = []
    for f in feats:
        top_r, bot_r = [], []
        for p in sorted(df[pcol].unique()):
            m = df[pcol] == p
            x, y = df.loc[m, f], df.loc[m, target]
            v = x.notna() & y.notna()
            if v.sum() < 8: continue
            xv, yv = x[v], y[v]
            top_r.extend(yv[xv >= xv.quantile(hi)].tolist())
            bot_r.extend(yv[xv <= xv.quantile(lo)].tolist())
        if len(top_r) < 5 or len(bot_r) < 5: continue
        ta, ba = np.array(top_r), np.array(bot_r)
        rows.append(dict(feature=f, top_mean=ta.mean(), top_hit=(ta>0).mean(), top_n=len(ta),
                         bot_mean=ba.mean(), bot_hit=(ba>0).mean(), bot_n=len(ba),
                         spread=ta.mean()-ba.mean(), hit_spread=(ta>0).mean()-(ba>0).mean()))
    return pd.DataFrame(rows).sort_values("spread", ascending=False, key=abs).reset_index(drop=True)


def compute_walk_forward(df, feats, target, pcol, min_train=4):
    periods = sorted(df[pcol].unique())
    recs = []
    for ti in range(min_train, len(periods)):
        m = df[pcol] == periods[ti]
        y = df.loc[m, target]
        row = {"period": periods[ti]}
        for f in feats: row[f] = _corr(df.loc[m, f], y, "spearman")
        recs.append(row)
    return pd.DataFrame(recs).set_index("period")


def compute_cumulative_dd(panel):
    cum = panel.cumsum()
    rows = []
    for f in cum.columns:
        ts = cum[f].dropna()
        if len(ts) < 2:
            rows.append(dict(feature=f, max_dd=0, dd_start=None, dd_end=None, dd_length=0))
            continue
        pk = ts.expanding().max(); dd = ts - pk; mdd = dd.min()
        if mdd < 0:
            de = dd.idxmin(); ds = ts.loc[:de].idxmax()
            dl = list(ts.index).index(de) - list(ts.index).index(ds)
        else: ds = de = None; dl = 0
        rows.append(dict(feature=f, max_dd=mdd, dd_start=ds, dd_end=de, dd_length=dl))
    return cum, pd.DataFrame(rows).set_index("feature")


def compute_pooled(df, feats, target, pcol, resid=True):
    d = df.copy()
    tgt = target
    if resid:
        d["_r"] = d[target] - d.groupby(pcol)[target].transform("mean")
        tgt = "_r"
    rows = []
    for f in feats:
        x, y = d[f], d[tgt]
        v = x.notna() & y.notna()
        if v.sum() < 20: continue
        rho, ps = spearmanr(x[v], y[v])
        r, pp = pearsonr(x[v], y[v])
        rows.append(dict(feature=f, pooled_spearman=rho, sp_p=ps,
                         pooled_pearson=r, pe_p=pp, n=v.sum()))
    return pd.DataFrame(rows).sort_values("pooled_spearman", ascending=False, key=abs).reset_index(drop=True)


def compute_partial(df, feats, target, pcol):
    pooled = compute_pooled(df, feats, target, pcol)
    if len(pooled) < 2: return pd.DataFrame()
    top_f = pooled.iloc[0]["feature"]
    rows = []
    for _, pr in pooled.iterrows():
        f = pr["feature"]
        if f == top_f:
            rows.append(dict(feature=f, standalone=pr["pooled_spearman"],
                             partial=pr["pooled_spearman"], ctrl="—"))
            continue
        x = df[f].rank(pct=True); z = df[top_f].rank(pct=True)
        y = df[target] - df.groupby(pcol)[target].transform("mean")
        v = x.notna() & z.notna() & y.notna()
        if v.sum() < 20: continue
        xv, zv, yv = x[v].values, z[v].values, y[v].values
        resid = xv - np.polyval(np.polyfit(zv, xv, 1), zv)
        rows.append(dict(feature=f, standalone=pr["pooled_spearman"],
                         partial=spearmanr(resid, yv)[0], ctrl=top_f))
    return pd.DataFrame(rows).sort_values("partial", ascending=False, key=abs).reset_index(drop=True)


def compute_segments(df, feats, target, pcol, scol):
    if scol not in df.columns: return pd.DataFrame()
    parts = []
    for seg in df[scol].dropna().unique():
        p = compute_pooled(df[df[scol] == seg], feats, target, pcol)
        p["segment"] = seg; parts.append(p)
    if not parts: return pd.DataFrame()
    c = pd.concat(parts, ignore_index=True)
    return c.pivot_table(index="feature", columns="segment", values="pooled_spearman").reindex(
        c.groupby("feature")["pooled_spearman"].apply(lambda x: x.abs().max())
        .sort_values(ascending=False).index)


# ═══════════════════════════════════════════════════════════════════
# PLOTTING — academic, plt.show()
# ═══════════════════════════════════════════════════════════════════

def _show(fig):
    fig.tight_layout(); plt.show(); plt.close(fig)


def plot_01_summary(s_sp, s_pe, top_n=25):
    d = s_sp.head(top_n); feats = d["feature"].values
    pe = s_pe.set_index("feature")["mean"].to_dict()
    fig, ax = plt.subplots(figsize=(9, max(5, len(feats)*0.3)))
    y = np.arange(len(feats)); sp = d["mean"].values
    ax.barh(y, sp, height=0.6, color=[POS if v>0 else NEG for v in sp], alpha=0.7, label="Spearman")
    ax.scatter([pe.get(f,0) for f in feats], y, c="black", s=18, zorder=5, marker="D", label="Pearson")
    ax.axvline(0, c="black", lw=0.5)
    for i, row in d.iterrows():
        sig = "***" if row["bh_pvalue"]<0.01 else "**" if row["bh_pvalue"]<0.05 else "*" if row["bh_pvalue"]<0.10 else ""
        ax.annotate(f'IR={row["ir"]:.2f} {sig}', xy=(row["mean"], i),
                    xytext=(4 if row["mean"]>=0 else -4, 0), textcoords="offset points",
                    fontsize=6, ha="left" if row["mean"]>=0 else "right", va="center")
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("Mean Cross-Sectional Correlation")
    ax.set_title("Figure 1: Spearman ρ (bars) vs Pearson r (◆)\nBH-adjusted: *p<.10 **p<.05 ***p<.01", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x"); _show(fig)


def plot_02_timeseries(panel, s_sp, top_n=6):
    top = [f for f in s_sp.head(top_n)["feature"] if f in panel.columns]
    fig, axes = plt.subplots(len(top), 1, figsize=(11, 2*len(top)), sharex=True)
    if len(top) == 1: axes = [axes]
    for i, f in enumerate(top):
        ax = axes[i]; ts = panel[f].dropna()
        ax.bar(range(len(ts)), ts.values, color=[POS if v>0 else NEG for v in ts.values], alpha=0.6, width=0.8)
        if len(ts)>3: ax.plot(range(len(ts)), ts.rolling(4,min_periods=2).mean().values, c="black", lw=1.2, label="4Q rolling")
        ax.axhline(0, c="black", lw=0.4)
        m,s = ts.mean(), ts.std(); ir = m/s if s>1e-12 else 0
        ax.set_title(f"{f}  mean={m:.3f} std={s:.3f} IR={ir:.2f}", fontsize=9, loc="left")
        ax.set_ylabel("ρ"); ax.legend(fontsize=7); ax.grid(axis="y")
        if i==len(top)-1:
            ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts.index, rotation=45, fontsize=6)
    fig.suptitle("Figure 2: Per-Period Spearman ρ", fontsize=11, y=1.01); _show(fig)


def plot_03_quintiles(qret, s_sp, top_n=8):
    top = [f for f in s_sp.head(top_n)["feature"] if f in qret][:top_n]
    nc = min(4, len(top)); nr = max(1, (len(top)+nc-1)//nc)
    fig, axes = plt.subplots(nr, nc, figsize=(3.5*nc, 3.2*nr))
    axes = np.array(axes).flatten() if len(top)>1 else [axes]
    for i, f in enumerate(top):
        ax = axes[i]; avg = qret[f].groupby("quintile")["mean_ret"].mean()
        hit = qret[f].groupby("quintile")["hit_rate"].mean()
        vals = [avg.get(q,0) for q in range(1,6)]; hits = [hit.get(q,.5) for q in range(1,6)]
        ax.bar(range(5), vals, color=QC, alpha=0.75, width=0.65)
        ax.set_xticks(range(5)); ax.set_xticklabels(["Q1","Q2","Q3","Q4","Q5"], fontsize=7)
        ax.axhline(0, c="black", lw=0.4)
        ax.set_title(f"{f}\nspread={vals[4]-vals[0]:.3f}", fontsize=8)
        ax.set_ylabel("Avg Δ%", fontsize=7); ax.grid(axis="y")
        for j,(v,h) in enumerate(zip(vals,hits)):
            ax.annotate(f'{h:.0%}', xy=(j,v), xytext=(0, 3 if v>=0 else -9),
                        textcoords="offset points", fontsize=6, ha="center",
                        color=POS if h>.55 else NEG if h<.45 else NEUT)
    for j in range(len(top), len(axes)): axes[j].set_visible(False)
    fig.suptitle("Figure 3: Quintile Analysis (labels = hit rate)", fontsize=11); _show(fig)


def plot_04_conditional(cond, top_n=20):
    d = cond.head(top_n); feats = d["feature"].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, max(4, len(feats)*0.28)))
    y = np.arange(len(feats)); w = 0.35
    a1.barh(y-w/2, d["top_mean"], w, color=POS, alpha=.7, label="Top Q")
    a1.barh(y+w/2, d["bot_mean"], w, color=NEG, alpha=.7, label="Bot Q")
    a1.set_yticks(y); a1.set_yticklabels(feats, fontsize=7); a1.axvline(0, c="black", lw=.4)
    a1.set_xlabel("Mean ΔPrice (%)"); a1.set_title("Conditional Mean Return")
    a1.legend(fontsize=7); a1.invert_yaxis(); a1.grid(axis="x")
    a2.barh(y-w/2, d["top_hit"]*100, w, color=POS, alpha=.7, label="Top Q")
    a2.barh(y+w/2, d["bot_hit"]*100, w, color=NEG, alpha=.7, label="Bot Q")
    a2.axvline(50, c="black", lw=.5, ls="--")
    a2.set_yticks(y); a2.set_yticklabels(feats, fontsize=7)
    a2.set_xlabel("Hit Rate (%)"); a2.set_title("Conditional Hit Rate")
    a2.legend(fontsize=7); a2.invert_yaxis(); a2.grid(axis="x")
    fig.suptitle("Figure 4: Top vs Bottom Quartile Returns", fontsize=11); _show(fig)


def plot_05_walkforward(wf, wf_s, top_n=6):
    top = [f for f in wf_s.head(top_n)["feature"] if f in wf.columns]
    fig, axes = plt.subplots(len(top), 1, figsize=(11, 1.8*len(top)), sharex=True)
    if len(top)==1: axes=[axes]
    for i, f in enumerate(top):
        ax = axes[i]; ts = wf[f].dropna()
        ax.bar(range(len(ts)), ts.values, color=[POS if v>0 else NEG for v in ts.values], alpha=.6, width=.8)
        ax.axhline(0, c="black", lw=.4); m,s=ts.mean(),ts.std(); ir=m/s if s>1e-12 else 0
        ax.set_title(f"[OOS] {f}  mean={m:.3f} IR={ir:.2f}", fontsize=9, loc="left")
        ax.set_ylabel("ρ"); ax.grid(axis="y")
        if i==len(top)-1:
            ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts.index, rotation=45, fontsize=6)
    fig.suptitle("Figure 5: Walk-Forward Out-of-Sample Spearman", fontsize=11, y=1.01); _show(fig)


def plot_06_cumulative(cum, dd, s_sp, top_n=8):
    top = [f for f in s_sp.head(top_n)["feature"] if f in cum.columns]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, f in enumerate(top):
        ts = cum[f].dropna(); ddv = dd.loc[f,"max_dd"] if f in dd.index else 0
        ax.plot(range(len(ts)), ts.values, color=PAL[i%len(PAL)], lw=1.3, label=f'{f} (dd={ddv:.2f})')
    ax.axhline(0, c="black", lw=.4)
    ax.set_xticks(range(len(cum))); ax.set_xticklabels(cum.index, rotation=45, fontsize=6)
    ax.set_ylabel("Cumulative Spearman ρ")
    ax.set_title("Figure 6: Cumulative Spearman (legend = max drawdown)")
    ax.legend(fontsize=7, ncol=2); ax.grid(); _show(fig)


def plot_07_sp_vs_pe(s_sp, s_pe, top_n=30):
    sp = s_sp.set_index("feature")["mean"].to_dict()
    pe = s_pe.set_index("feature")["mean"].to_dict()
    common = [f for f in sp if f in pe][:top_n]
    fig, ax = plt.subplots(figsize=(7, 7))
    sx = [sp[f] for f in common]; px = [pe[f] for f in common]
    ax.scatter(sx, px, c=POS, s=28, alpha=.7, edgecolors=NEUT, lw=.5)
    lim = max(max(abs(v) for v in sx+px), .05)*1.3
    ax.plot([-lim,lim],[-lim,lim], c="black", ls="--", lw=.7, label="y=x")
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    for f,s,p in zip(common,sx,px):
        if abs(s)>.02 or abs(p)>.02: ax.annotate(f, (s,p), fontsize=5.5, xytext=(2,2), textcoords="offset points")
    ax.set_xlabel("Mean Spearman ρ"); ax.set_ylabel("Mean Pearson r")
    ax.set_title("Figure 7: Spearman vs Pearson\nAbove diag → convex; Below → rank-only")
    ax.legend(fontsize=8); ax.grid(); ax.set_aspect("equal"); _show(fig)


def plot_08_heatmap(panel, s_sp, window=4, top_n=15):
    top = [f for f in s_sp.head(top_n)["feature"] if f in panel.columns]
    roll = panel[top].rolling(window, min_periods=2).mean()
    fig, ax = plt.subplots(figsize=(12, max(4, len(top)*.35)))
    im = ax.imshow(roll.T.values, aspect="auto", cmap="RdBu_r", vmin=-.25, vmax=.25)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7)
    ax.set_xticks(range(len(roll))); ax.set_xticklabels(roll.index, rotation=45, fontsize=6)
    ax.set_title(f"Figure 8: Rolling {window}-Period Spearman Heatmap")
    plt.colorbar(im, ax=ax, label="ρ", shrink=.8); _show(fig)


def plot_09_corr(df, feats, top_n=20):
    fs = feats[:top_n]; c = df[fs].rank(pct=True).corr("spearman")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(fs))); ax.set_xticklabels(fs, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(fs))); ax.set_yticklabels(fs, fontsize=6)
    for i in range(len(fs)):
        for j in range(len(fs)):
            v = c.values[i,j]
            if abs(v)>.3 and i!=j: ax.text(j,i,f"{v:.2f}", ha="center", va="center", fontsize=5)
    ax.set_title("Figure 9: Feature Rank Correlation (high ⇒ redundant)")
    plt.colorbar(im, ax=ax, shrink=.8); _show(fig)


def plot_10_partial(part, top_n=20):
    d = part.head(top_n); feats = d["feature"].values
    fig, ax = plt.subplots(figsize=(9, max(4, len(feats)*.28)))
    y = np.arange(len(feats)); w = .35
    ax.barh(y-w/2, d["standalone"], w, color=POS, alpha=.7, label="Standalone")
    ax.barh(y+w/2, d["partial"], w, color=NEG, alpha=.7, label="Partial")
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=7); ax.axvline(0, c="black", lw=.4)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Figure 10: Marginal Contribution (partial = residualized)")
    ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(axis="x"); _show(fig)


def plot_11_segments(seg, top_n=20):
    if seg.empty: return
    d = seg.head(top_n); feats = d.index.tolist(); segs = d.columns.tolist()
    fig, ax = plt.subplots(figsize=(9, max(4, len(feats)*.28)))
    y = np.arange(len(feats)); w = .7/len(segs)
    for si, s in enumerate(segs):
        off = (si - len(segs)/2 + .5) * w
        ax.barh(y+off, d[s].values, height=w, color=PAL[si], alpha=.75, label=s)
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=7); ax.axvline(0, c="black", lw=.4)
    ax.set_xlabel("Pooled Spearman ρ"); ax.set_title("Figure 11: Segment Analysis")
    ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(axis="x"); _show(fig)


def plot_12_dist(panel, s_sp, top_n=6):
    top = [f for f in s_sp.head(top_n)["feature"] if f in panel.columns]
    nc = min(3, len(top)); nr = max(1, (len(top)+nc-1)//nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 3*nr))
    axes = np.array(axes).flatten() if len(top)>1 else [axes]
    for i, f in enumerate(top):
        ax = axes[i]; vals = panel[f].dropna().values
        ax.hist(vals, bins=12, color=PAL[i%len(PAL)], alpha=.65, edgecolor="white")
        ax.axvline(0, c="black", lw=.5, ls="--")
        ax.axvline(vals.mean(), c=NEG, lw=1.5, label=f"μ={vals.mean():.3f}")
        ax.set_title(f, fontsize=9); ax.set_xlabel("ρ"); ax.legend(fontsize=7); ax.grid(axis="y")
    for j in range(len(top), len(axes)): axes[j].set_visible(False)
    fig.suptitle("Figure 12: Distribution of Per-Period Spearman ρ", fontsize=11); _show(fig)


# ═══════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════

class Pipeline:
    """
    Everything numeric not in meta_cols / target = feature.
    All results stored as DataFrames after run().
    """

    def __init__(self, df, target="price_change", period_col="cal_quarter",
                 meta_cols=None, segment_col="ann_type"):
        self.df = df.copy()
        self.target = target
        self.period_col = period_col
        self.segment_col = segment_col
        exclude = set(meta_cols or []); exclude.add(target)
        self.features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

        # Results — all None until run()
        self.spearman_panel = self.pearson_panel = None
        self.summary_spearman = self.summary_pearson = None
        self.pooled = self.conditional = None
        self.quintile_returns_raw = self.quintile_summary = None
        self.walk_forward_panel = self.wf_summary = None
        self.cumulative = self.drawdowns = None
        self.partial = self.segments = None

    def run(self, min_train=4, rolling_window=4, plot=True):
        df, feats, tgt, pcol = self.df, self.features, self.target, self.period_col
        nf = len(feats)

        print("=" * 64)
        print("  EARNINGS SIGNAL PIPELINE")
        print("=" * 64)
        t = df[tgt]
        print(f"  {len(df)} obs | {df[pcol].nunique()} periods | {nf} features")
        print(f"  Target: {tgt}  μ={t.mean():.4f}  σ={t.std():.4f}  skew={t.skew():.3f}")
        print(f"\n  Features ({nf}):")
        for i in range(0, nf, 6): print(f"    {', '.join(feats[i:i+6])}")

        print("\n[1/11] Spearman panel...")
        self.spearman_panel = compute_corr_panel(df, feats, tgt, pcol, "spearman")

        print("[2/11] Pearson panel...")
        self.pearson_panel = compute_corr_panel(df, feats, tgt, pcol, "pearson")

        print("[3/11] Summary + BH...")
        self.summary_spearman = compute_summary(self.spearman_panel, nf)
        self.summary_pearson = compute_summary(self.pearson_panel, nf)
        _pt("Spearman Summary", self.summary_spearman)

        print("[4/11] Pooled (residualized)...")
        self.pooled = compute_pooled(df, feats, tgt, pcol)
        _pt("Pooled", self.pooled, 15)

        print("[5/11] Conditional returns...")
        self.conditional = compute_conditional(df, feats, tgt, pcol)
        _pt("Conditional", self.conditional, 15)

        print("[6/11] Quintiles...")
        self.quintile_returns_raw = compute_quintile_returns(df, feats, tgt, pcol)
        self.quintile_summary = compute_quintile_summary(self.quintile_returns_raw)

        print(f"[7/11] Walk-forward (min_train={min_train})...")
        self.walk_forward_panel = compute_walk_forward(df, feats, tgt, pcol, min_train)
        self.wf_summary = compute_summary(self.walk_forward_panel, nf)
        _pt("Walk-Forward OOS", self.wf_summary, 15)

        print("[8/11] Cumulative + drawdown...")
        self.cumulative, self.drawdowns = compute_cumulative_dd(self.spearman_panel)

        print("[9/11] Partial...")
        self.partial = compute_partial(df, feats, tgt, pcol)
        _pt("Partial", self.partial, 15)

        print("[10/11] Segments...")
        self.segments = pd.DataFrame()
        if self.segment_col and self.segment_col in df.columns:
            self.segments = compute_segments(df, feats, tgt, pcol, self.segment_col)

        if plot:
            print("[11/11] Plotting...")
            ordered = self.summary_spearman["feature"].tolist()
            plot_01_summary(self.summary_spearman, self.summary_pearson)
            plot_02_timeseries(self.spearman_panel, self.summary_spearman)
            plot_03_quintiles(self.quintile_returns_raw, self.summary_spearman)
            plot_04_conditional(self.conditional)
            plot_05_walkforward(self.walk_forward_panel, self.wf_summary)
            plot_06_cumulative(self.cumulative, self.drawdowns, self.summary_spearman)
            plot_07_sp_vs_pe(self.summary_spearman, self.summary_pearson)
            plot_08_heatmap(self.spearman_panel, self.summary_spearman, rolling_window)
            plot_09_corr(df, ordered)
            plot_10_partial(self.partial)
            if not self.segments.empty: plot_11_segments(self.segments)
            plot_12_dist(self.spearman_panel, self.summary_spearman)

        print("\n" + "=" * 64)
        print("  COMPLETE — results: pipe.summary_spearman, pipe.pooled, etc.")
        print("=" * 64)
        return self


def _pt(title, df, n=20):
    print(f"\n  ── {title} ──")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_rows", n, "display.width", 140):
        print(df.head(n).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_sample(n_q=20, n_t=40, seed=42):
    rng = np.random.RandomState(seed)
    from scipy.stats import beta as bd
    tks = [("AAPL","Apple","Tech","HW"),("MSFT","MSFT","Tech","SW"),("GOOGL","Alphabet","Tech","SW"),
           ("AMZN","Amazon","CD","Retail"),("NVDA","NVIDIA","Tech","Semi"),("META","Meta","Tech","SW"),
           ("TSLA","Tesla","CD","Auto"),("JPM","JPM","Fin","Bank"),("JNJ","JNJ","HC","Pharma"),
           ("V","Visa","Fin","AM"),("UNH","UNH","HC","MT"),("HD","HD","CD","Retail"),
           ("PG","PG","CD","Retail"),("MA","MA","Fin","AM"),("LLY","Lilly","HC","Pharma"),
           ("ABBV","AbbVie","HC","Bio"),("MRK","Merck","HC","Pharma"),("AVGO","Broadcom","Tech","Semi"),
           ("PEP","Pepsi","CD","Retail"),("COST","Costco","CD","Retail"),
           ("TMO","Thermo","HC","MT"),("ADBE","Adobe","Tech","SW"),("CRM","CRM","Tech","SW"),
           ("AMD","AMD","Tech","Semi"),("INTC","Intel","Tech","Semi"),("BA","Boeing","Ind","Aero"),
           ("CAT","CAT","Ind","Mach"),("GS","GS","Fin","Bank"),("MS","MS","Fin","Bank"),
           ("BLK","BLK","Fin","AM"),("ISRG","ISRG","HC","MT"),("RTX","RTX","Ind","Aero"),
           ("DAL","Delta","Ind","Trans"),("UAL","United","Ind","Trans"),("WFC","WFC","Fin","Bank"),
           ("C","Citi","Fin","Bank"),("PANW","PANW","Tech","SW"),("NOW","NOW","Tech","SW"),
           ("UBER","Uber","Tech","SW"),("AXP","AmEx","Fin","Bank")][:n_t]
    sb = {"Tech":.3,"HC":.1,"Fin":-.05,"CD":.05,"Ind":0}
    METRICS = ["revenue","eps","ebitda","ebit","net_income","gross_profit","operating_income",
               "gross_margin","operating_margin","net_margin","fcf","capex","total_debt","net_debt",
               "total_assets","book_value","roe","roa","roic","div_ps","buyback","op_cf","inv_cf",
               "fin_cf","rev_growth","eps_growth","sga","rd","cogs","tax_rate"]
    qs = [(2021+q//4, q%4+1) for q in range(n_q)]
    rows = []
    for yr, qn in qs:
        ql = f"{yr}Q{qn}"; am = min((qn-1)*3+4+rng.randint(0,2),12)
        itr = {}; order = list(range(len(tks))); rng.shuffle(order)
        for idx in order:
            tk, nm, sec, ind = tks[idx]
            ad = 5+rng.randint(0,24)
            row = dict(ticker=tk, name=nm, sector=sec, industry=ind,
                       ann_date=f"{yr}-{am:02d}-{ad:02d}", ann_type=rng.choice(["AMC","BMO"]),
                       cal_quarter=ql)
            for m in METRICS:
                if "margin" in m or m in ("roe","roa","roic","tax_rate"): base=10+rng.rand()*30
                elif "growth" in m: base=-5+rng.rand()*25
                elif m=="eps": base=.5+rng.rand()*4
                elif m=="revenue": base=1000+rng.rand()*50000
                else: base=50+rng.rand()*5000
                act=base*(1+rng.randn()*.08); est=base*(1+rng.randn()*.03)
                row[m]=round(act,4); row[f"est_{m}"]=round(est,4)
                row[f"surprise_{m}"]=round(((act-est)/max(abs(est),1e-8))*100,4)
            rd=rng.randn()*3
            for w in [7,14,30,60,90]: row[f"eps_rev_{w}d"]=round(rd+rng.randn()*1.5,4)
            row["eps_rev_vol"]=round(abs(rng.randn())*2,4)
            row["eps_rev_breadth"]=round((rng.rand()-.3)*2,4)
            ik=f"{ind}_{ql}"
            if ik not in itr: itr[ik]=dict(tot=sum(1 for t in tks if t[3]==ind),pos=0,neg=0,rep=0)
            tr=itr[ik]; a,b=tr["pos"]+1,tr["neg"]+1; tau=.5
            sep=round(1-bd.cdf(tau/(1+tau),a,b),4)
            row.update(ind_total=tr["tot"],ind_reported=tr["rep"],ind_pos=tr["pos"],
                       ind_neg=tr["neg"],tau=tau,spread_exceed_prob=sep)
            sig=(sb.get(sec,0)+(.3 if row["eps_rev_30d"]>0 else -.2)*.5
                 +(.2 if sep>.5 else -.15)*.3+(.5 if row["surprise_eps"]>0 else -.4)*1.2+rng.randn()*2.5)
            row["price_change"]=round(sig,4)
            if row["price_change"]>0: tr["pos"]+=1
            else: tr["neg"]+=1
            tr["rep"]+=1; rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = generate_sample(n_q=20, seed=42)

    # Add composites — just new columns
    df["rev_x_spread"] = df["eps_rev_30d"] * df["spread_exceed_prob"]
    df["rev_accel"] = df["eps_rev_7d"] - df["eps_rev_30d"]
    df["rev_snr"] = df["eps_rev_30d"] / df["eps_rev_vol"].clip(lower=0.01)

    meta = ["ticker", "name", "sector", "industry", "ann_date", "ann_type", "cal_quarter"]

    pipe = Pipeline(df, target="price_change", period_col="cal_quarter",
                    meta_cols=meta, segment_col="ann_type")
    pipe.run()

    # Afterwards, all results are DataFrames:
    # pipe.summary_spearman.head(20)
    # pipe.pooled.query("sp_p < 0.05")
    # pipe.quintile_summary.sort_values("monotonicity", ascending=False)
    # pipe.walk_forward_panel["eps_rev_30d"].cumsum().plot()
    # pipe.drawdowns.sort_values("max_dd")
    # pipe.partial.head(10)