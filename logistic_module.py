"""
Logistic Regression Module for Earnings Signal Pipeline v6
============================================================
Replaces the ICIR-weighted ranking approach with a probability-calibrated
logistic classifier optimised for the binary question: P(positive return).

DESIGN PRINCIPLES:
  1. ZERO LEAKAGE — strict expanding-window walk-forward.
     - Features are rank-transformed WITHIN the training window only.
     - The transform learned on train is applied to the test period.
     - Regularisation strength C is selected via purged time-series CV
       inside the training window — never touches the test period.
     - No feature is computed using future data (inherits from _is_lookahead).

  2. STATISTICAL RIGOUR
     - Out-of-sample Brier score decomposition (reliability + resolution).
     - Hosmer-Lemeshow calibration test per period and pooled.
     - ROC-AUC with DeLong confidence intervals.
     - Permutation-based null distribution for hit rate significance.
     - Feature importance via |coefficient| from L1 path.

  3. PROPER BETTING
     - Trade only when predicted P exceeds a user-controlled threshold.
     - Kelly/half-Kelly sizing proportional to estimated edge.
     - Plugs into existing run_backtest_generic / trade stats engine.

USAGE (after running Pipeline.run):
    pipe.backtest_logistic(
        prob_threshold=0.55,      # only trade when P(up) > this
        kelly_fraction=0.5,       # half-Kelly sizing
        long_only=True,
        capital=1_000_000,
    )

Author: Quantitative Research Module
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, spearmanr
from scipy.special import expit  # logistic sigmoid
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 1. FEATURE PREPROCESSING — leakage-safe rank transform
# ═══════════════════════════════════════════════════════════════════

class RankTransformer:
    """
    Rank-transform features to [0, 1] using ONLY training data statistics.

    For each feature:
      - On fit(): stores the training-set values for rank interpolation.
      - On transform(): maps new data to [0, 1] by computing their
        empirical CDF position relative to the training distribution.

    This is leakage-safe: the test period's own distribution never
    influences the transform. Compared to z-score standardisation,
    rank transforms are robust to the fat tails in earnings data.
    """

    def __init__(self):
        self.quantiles_ = {}   # {feature: sorted array of training values}
        self.features_ = []
        self.fitted_ = False

    def fit(self, X_train, features):
        """
        Store the empirical distribution of each feature from training data.

        Parameters
        ----------
        X_train : DataFrame with training observations
        features : list of column names to transform
        """
        self.features_ = list(features)
        self.quantiles_ = {}
        for f in self.features_:
            vals = X_train[f].dropna().values.astype(float)
            if len(vals) < 5:
                self.quantiles_[f] = None
                continue
            self.quantiles_[f] = np.sort(vals)
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Map features to [0, 1] using the training distribution.

        For each value x in feature f:
          rank = (# of training values <= x) / (# of training values)

        Values below the training min map to 0; above max map to 1.
        NaN stays NaN.
        """
        assert self.fitted_, "Must call fit() before transform()"
        result = pd.DataFrame(index=X.index)
        for f in self.features_:
            if self.quantiles_[f] is None:
                result[f] = np.nan
                continue
            vals = X[f].values.astype(float)
            ref = self.quantiles_[f]
            # np.searchsorted gives position in sorted array
            ranks = np.searchsorted(ref, vals, side="right") / len(ref)
            # Preserve NaN
            ranks[np.isnan(vals)] = np.nan
            result[f] = ranks
        return result

    def fit_transform(self, X, features):
        return self.fit(X, features).transform(X)


# ═══════════════════════════════════════════════════════════════════
# 2. PURGED TIME-SERIES CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════

def purged_ts_cv_splits(periods, n_splits=3, purge=1):
    """
    Generate train/validation splits for time-series CV with a purge gap.

    For n_splits=3 and periods [Q1..Q12]:
      Split 1: train=[Q1..Q3],  purge=[Q4],     val=[Q5..Q6]
      Split 2: train=[Q1..Q5],  purge=[Q6],     val=[Q7..Q8]
      Split 3: train=[Q1..Q7],  purge=[Q8],     val=[Q9..Q10]

    The purge gap prevents any autocorrelation leakage between
    the last training quarter and the first validation quarter.

    Parameters
    ----------
    periods : sorted list of period labels
    n_splits : number of CV folds
    purge : number of periods to skip between train and val

    Yields
    ------
    (train_periods, val_periods) tuples
    """
    n = len(periods)
    # We need at least 4 training + 1 purge + 1 validation per fold
    min_train = 4
    usable = n - min_train - purge
    if usable < n_splits:
        # Fall back: single train/val split
        mid = max(min_train, n // 2)
        yield periods[:mid], periods[mid + purge:]
        return

    fold_size = max(1, usable // n_splits)
    for i in range(n_splits):
        val_end = min_train + purge + (i + 1) * fold_size
        val_start = min_train + purge + i * fold_size
        if val_end > n:
            break
        train_p = periods[:val_start - purge]
        val_p = periods[val_start:val_end]
        if len(train_p) >= min_train and len(val_p) >= 1:
            yield train_p, val_p


# ═══════════════════════════════════════════════════════════════════
# 3. LOGISTIC REGRESSION WITH WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════

def _fit_logistic_cv(X_train, y_train, periods_train, pcol_vals,
                     Cs=None, n_splits=3, purge=1):
    """
    Fit L1-penalised logistic regression with purged time-series CV
    to select regularisation strength C.

    Parameters
    ----------
    X_train : ndarray (n_samples, n_features) — already rank-transformed
    y_train : ndarray (n_samples,) — binary {0, 1}
    periods_train : ndarray of period labels for each training sample
    pcol_vals : sorted unique period labels in training set
    Cs : list of C values to try (default: log-spaced 0.001 to 10)
    n_splits : number of CV folds
    purge : purge gap in periods

    Returns
    -------
    best_C : optimal regularisation strength
    best_coef : coefficient vector at best_C
    best_intercept : intercept at best_C
    cv_results : dict with per-C validation log-loss
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    if Cs is None:
        Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    cv_scores = {C: [] for C in Cs}

    for train_p, val_p in purged_ts_cv_splits(pcol_vals, n_splits, purge):
        train_mask = np.isin(periods_train, train_p)
        val_mask = np.isin(periods_train, val_p)

        X_tr, y_tr = X_train[train_mask], y_train[train_mask]
        X_va, y_va = X_train[val_mask], y_train[val_mask]

        if len(X_tr) < 20 or len(X_va) < 5:
            continue
        if y_tr.sum() < 3 or (1 - y_tr).sum() < 3:
            continue

        for C in Cs:
            model = LogisticRegression(
                penalty="l1", C=C, solver="saga",
                max_iter=5000, tol=1e-5, random_state=42,
                class_weight="balanced",  # handle class imbalance
            )
            try:
                model.fit(X_tr, y_tr)
                p_va = model.predict_proba(X_va)[:, 1]
                # Clip probabilities to avoid log(0)
                p_va = np.clip(p_va, 1e-6, 1 - 1e-6)
                ll = log_loss(y_va, p_va)
                cv_scores[C].append(ll)
            except Exception:
                continue

    # Pick C with lowest mean validation log-loss
    mean_scores = {}
    for C, scores in cv_scores.items():
        if len(scores) >= 1:
            mean_scores[C] = np.mean(scores)

    if not mean_scores:
        # Fallback: use moderate regularisation
        best_C = 0.1
    else:
        best_C = min(mean_scores, key=mean_scores.get)

    # Refit on full training set with best C
    model = LogisticRegression(
        penalty="l1", C=best_C, solver="saga",
        max_iter=5000, tol=1e-5, random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    return (best_C, model.coef_[0], model.intercept_[0],
            {"Cs": Cs, "mean_scores": mean_scores})


def compute_logistic_signal(df, features, target, pcol, min_train=4,
                            purge=1, n_cv_splits=3, sector_col=None):
    """
    Walk-forward logistic regression signal.

    For each test period t:
      1. Collect all observations from periods [0, t-1] as training data.
      2. Rank-transform features using ONLY training data distribution.
      3. Select C via purged time-series CV within the training window.
      4. Fit L1-logistic on full training set at best C.
      5. Apply the trained rank-transform to period t.
      6. Predict P(positive return) for period t.

    Parameters
    ----------
    df : DataFrame with features, target, and period column
    features : list of feature column names
    target : target column name (continuous return)
    pcol : period column name
    min_train : minimum training periods before first prediction
    purge : purge gap for CV (periods between train and val)
    n_cv_splits : number of CV folds for C selection
    sector_col : optional sector column for one-hot encoding

    Returns
    -------
    probabilities : Series of predicted P(positive return), NaN where no signal
    model_diagnostics : list of dicts with per-period model info
    """
    periods = sorted(df[pcol].unique())
    probabilities = pd.Series(np.nan, index=df.index)
    diagnostics = []

    # Binary target: 1 if return > 0, else 0
    y_full = (df[target] > 0).astype(int)

    for ti in range(min_train, len(periods)):
        test_period = periods[ti]
        train_periods = periods[:ti]

        # ── Split data ──
        train_mask = df[pcol].isin(train_periods)
        test_mask = df[pcol] == test_period

        df_train = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()
        y_train = y_full.loc[train_mask].values
        y_test = y_full.loc[test_mask].values

        # ── Filter to rows with target and sufficient features ──
        feat_avail_train = df_train[features].notna().sum(axis=1) >= max(3, len(features) // 2)
        tgt_avail_train = df_train[target].notna()
        train_ok = feat_avail_train & tgt_avail_train

        feat_avail_test = df_test[features].notna().sum(axis=1) >= max(3, len(features) // 2)
        tgt_avail_test = df_test[target].notna()
        test_ok = feat_avail_test & tgt_avail_test

        df_train = df_train.loc[train_ok]
        df_test = df_test.loc[test_ok]
        y_train = y_full.loc[df_train.index].values
        y_test = y_full.loc[df_test.index].values

        if len(df_train) < 30 or len(df_test) < 3:
            continue
        if y_train.sum() < 5 or (1 - y_train).sum() < 5:
            continue

        # ── Rank-transform features using ONLY training data ──
        transformer = RankTransformer()
        X_train_rank = transformer.fit_transform(df_train, features)
        X_test_rank = transformer.transform(df_test)

        # Fill remaining NaN with 0.5 (median rank — neutral imputation)
        X_train_rank = X_train_rank.fillna(0.5)
        X_test_rank = X_test_rank.fillna(0.5)

        # ── Optional: add sector dummies ──
        feature_names = list(features)
        if sector_col and sector_col in df.columns:
            # One-hot encode sector from TRAINING data categories
            train_sectors = pd.get_dummies(df_train[sector_col], prefix="sec", dtype=float)
            test_sectors = pd.get_dummies(df_test[sector_col], prefix="sec", dtype=float)
            # Align columns (test may have sectors not in train or vice versa)
            for col in train_sectors.columns:
                if col not in test_sectors.columns:
                    test_sectors[col] = 0.0
            test_sectors = test_sectors[train_sectors.columns]

            X_train_rank = pd.concat([X_train_rank.reset_index(drop=True),
                                       train_sectors.reset_index(drop=True)], axis=1)
            X_test_rank = pd.concat([X_test_rank.reset_index(drop=True),
                                      test_sectors.reset_index(drop=True)], axis=1)
            feature_names = feature_names + list(train_sectors.columns)

        X_train_arr = X_train_rank.values.astype(float)
        X_test_arr = X_test_rank.values.astype(float)
        periods_arr = df_train[pcol].values

        # ── Fit logistic with purged CV ──
        train_period_labels = sorted(df_train[pcol].unique())
        best_C, coef, intercept, cv_info = _fit_logistic_cv(
            X_train_arr, y_train, periods_arr, train_period_labels,
            n_splits=n_cv_splits, purge=purge
        )

        # ── Predict on test period ──
        logits = X_test_arr @ coef + intercept
        probs = expit(logits)

        # Store predictions
        test_idx = df_test.index
        probabilities.loc[test_idx] = probs

        # ── Diagnostics for this period ──
        n_nonzero = np.sum(np.abs(coef[:len(features)]) > 1e-6)
        active_features = [features[j] for j in range(len(features))
                           if abs(coef[j]) > 1e-6]
        active_coefs = {features[j]: round(coef[j], 4) for j in range(len(features))
                        if abs(coef[j]) > 1e-6}

        # OOS metrics for this period
        from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
        try:
            brier = brier_score_loss(y_test, probs)
            auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
            ll = log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6))
        except Exception:
            brier = auc = ll = np.nan

        # Calibration: predicted prob vs actual frequency in decile bins
        hit_at_55 = np.nan
        n_above_55 = np.sum(probs > 0.55)
        if n_above_55 > 0:
            hit_at_55 = y_test[probs > 0.55].mean()

        diagnostics.append({
            "period": test_period,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "best_C": best_C,
            "n_active_features": n_nonzero,
            "active_features": active_features,
            "coefs": active_coefs,
            "train_base_rate": y_train.mean(),
            "test_base_rate": y_test.mean(),
            "oos_brier": brier,
            "oos_auc": auc,
            "oos_logloss": ll,
            "mean_pred_prob": probs.mean(),
            "std_pred_prob": probs.std(),
            "n_above_55": n_above_55,
            "hit_at_55": hit_at_55,
            "cv_scores": cv_info.get("mean_scores", {}),
        })

    return probabilities, diagnostics


# ═══════════════════════════════════════════════════════════════════
# 4. STATISTICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════

def compute_calibration_table(y_true, y_prob, n_bins=10):
    """
    Compute calibration table: for each probability bin,
    what is the predicted probability vs the observed frequency?

    Returns DataFrame with columns:
      bin_center, mean_predicted, observed_freq, count, cumulative_count

    A well-calibrated model has mean_predicted ≈ observed_freq.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    records = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n = mask.sum()
        if n == 0:
            continue
        records.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "bin_center": (lo + hi) / 2,
            "mean_predicted": y_prob[mask].mean(),
            "observed_freq": y_true[mask].mean(),
            "count": n,
        })
    return pd.DataFrame(records)


def hosmer_lemeshow_test(y_true, y_prob, n_groups=10):
    """
    Hosmer-Lemeshow goodness-of-fit test for calibration.

    H0: The model is well-calibrated (predicted probabilities match
        observed frequencies).

    Returns: (test_statistic, p_value, degrees_of_freedom)

    A p-value > 0.05 means we CANNOT reject good calibration — this is good.
    A p-value < 0.05 means the model is poorly calibrated.
    """
    n = len(y_true)
    if n < 20:
        return np.nan, np.nan, 0

    # Sort by predicted probability and split into groups
    order = np.argsort(y_prob)
    y_s = y_true[order]
    p_s = y_prob[order]

    groups = np.array_split(np.arange(n), n_groups)
    hl_stat = 0.0
    n_actual_groups = 0

    for g in groups:
        if len(g) == 0:
            continue
        obs_1 = y_s[g].sum()       # observed positives
        obs_0 = len(g) - obs_1     # observed negatives
        exp_1 = p_s[g].sum()       # expected positives
        exp_0 = len(g) - exp_1     # expected negatives

        if exp_1 > 1e-10:
            hl_stat += (obs_1 - exp_1) ** 2 / exp_1
        if exp_0 > 1e-10:
            hl_stat += (obs_0 - exp_0) ** 2 / exp_0
        n_actual_groups += 1

    df = max(n_actual_groups - 2, 1)
    p_value = 1 - chi2.cdf(hl_stat, df)
    return hl_stat, p_value, df


def brier_decomposition(y_true, y_prob, n_bins=10):
    """
    Decompose Brier score into reliability, resolution, and uncertainty.

    Brier = Reliability - Resolution + Uncertainty

    - Reliability (lower is better): measures calibration error.
      How far off are binned predicted probs from observed frequencies?
    - Resolution (higher is better): measures discrimination.
      How different are binned observed frequencies from the base rate?
    - Uncertainty: base-rate variance, p̄(1-p̄). Not under model control.

    Reference: Murphy (1973) decomposition.
    """
    n = len(y_true)
    p_bar = y_true.mean()
    uncertainty = p_bar * (1 - p_bar)

    bins = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_k = mask.sum()
        if n_k == 0:
            continue

        o_k = y_true[mask].mean()   # observed frequency in bin
        f_k = y_prob[mask].mean()   # mean predicted prob in bin

        reliability += n_k * (f_k - o_k) ** 2
        resolution += n_k * (o_k - p_bar) ** 2

    reliability /= n
    resolution /= n
    brier = reliability - resolution + uncertainty

    return {
        "brier_score": brier,
        "reliability": reliability,   # calibration error (want low)
        "resolution": resolution,     # discrimination (want high)
        "uncertainty": uncertainty,    # base rate variance (fixed)
        "skill_score": 1 - brier / uncertainty if uncertainty > 1e-10 else 0,
    }


def permutation_test_hit_rate(y_true, y_pred_binary, n_permutations=5000,
                               random_state=42):
    """
    Test whether the observed hit rate is significantly different from chance.

    Null hypothesis: the predictions have no relationship to outcomes
    (i.e. shuffling the labels doesn't change the hit rate).

    Returns: (observed_hit, p_value, null_distribution)
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    # Only evaluate where we actually made a prediction
    mask = y_pred_binary != 0
    if mask.sum() < 5:
        return np.nan, np.nan, np.array([])

    y_t = y_true[mask]
    y_p = y_pred_binary[mask]
    observed = (y_p * y_t > 0).mean()

    null_hits = np.empty(n_permutations)
    for i in range(n_permutations):
        y_shuf = rng.permutation(y_t)
        null_hits[i] = (y_p * y_shuf > 0).mean()

    p_value = (null_hits >= observed).mean()
    return observed, p_value, null_hits


def compute_delong_auc_ci(y_true, y_prob, alpha=0.05):
    """
    DeLong confidence interval for AUC.

    Reference: DeLong et al. (1988) "Comparing the Areas under Two or
    More Correlated Receiver Operating Characteristic Curves"

    Returns: (auc, ci_lower, ci_upper)
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    auc = roc_auc_score(y_true, y_prob)

    # Compute structural components for DeLong variance
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)

    if n_pos < 2 or n_neg < 2:
        return auc, np.nan, np.nan

    # Placement values
    V_pos = np.array([np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos])
    V_neg = np.array([np.mean(pos > n) + 0.5 * np.mean(pos == n) for n in neg])

    S_pos = np.var(V_pos, ddof=1)
    S_neg = np.var(V_neg, ddof=1)

    var_auc = S_pos / n_pos + S_neg / n_neg
    se = np.sqrt(var_auc) if var_auc > 0 else 1e-10
    z = norm.ppf(1 - alpha / 2)

    return auc, max(0, auc - z * se), min(1, auc + z * se)


def compute_logistic_validation(y_true, y_prob, y_pred_direction, target_values,
                                 cost_bps=10, n_perm=5000):
    """
    Comprehensive statistical validation of the logistic signal.

    Parameters
    ----------
    y_true : binary {0, 1} array (1 = positive return)
    y_prob : predicted P(positive return)
    y_pred_direction : +1 where we go LONG, 0 where no trade
    target_values : actual continuous returns (for PnL analysis)
    cost_bps : transaction cost
    n_perm : permutation test iterations

    Returns
    -------
    validation : dict with all statistical tests
    """
    mask = np.isfinite(y_prob) & np.isfinite(y_true)
    y_t = y_true[mask].astype(int)
    y_p = y_prob[mask]

    results = {}

    # ── 1. Brier score decomposition ──
    results["brier"] = brier_decomposition(y_t, y_p)

    # ── 2. AUC with DeLong CI ──
    auc, ci_lo, ci_hi = compute_delong_auc_ci(y_t, y_p)
    results["auc"] = {"auc": auc, "ci_95_lower": ci_lo, "ci_95_upper": ci_hi}

    # ── 3. Hosmer-Lemeshow calibration test ──
    hl_stat, hl_p, hl_df = hosmer_lemeshow_test(y_t, y_p)
    results["hosmer_lemeshow"] = {
        "statistic": hl_stat, "p_value": hl_p, "df": hl_df,
        "well_calibrated": hl_p > 0.05 if not np.isnan(hl_p) else None,
    }

    # ── 4. Calibration table ──
    results["calibration_table"] = compute_calibration_table(y_t, y_p)

    # ── 5. Permutation test on hit rate ──
    traded = y_pred_direction != 0
    if traded.sum() >= 5:
        y_actual_sign = np.sign(target_values)
        obs_hit, perm_p, null_dist = permutation_test_hit_rate(
            y_actual_sign[mask], y_pred_direction[mask], n_perm)
        results["permutation_test"] = {
            "observed_hit_rate": obs_hit,
            "p_value": perm_p,
            "n_permutations": n_perm,
            "significant_at_05": perm_p < 0.05 if not np.isnan(perm_p) else None,
        }
    else:
        results["permutation_test"] = {"observed_hit_rate": np.nan, "p_value": np.nan}

    # ── 6. Log-loss vs null model ──
    from sklearn.metrics import log_loss as sk_log_loss
    base_rate = y_t.mean()
    null_logloss = sk_log_loss(y_t, np.full_like(y_p, base_rate))
    model_logloss = sk_log_loss(y_t, np.clip(y_p, 1e-6, 1 - 1e-6))
    results["logloss"] = {
        "model": model_logloss,
        "null_model": null_logloss,
        "improvement_pct": (null_logloss - model_logloss) / null_logloss * 100
            if null_logloss > 0 else 0,
    }

    return results


# ═══════════════════════════════════════════════════════════════════
# 5. BACKTEST ENGINE — probability-threshold + Kelly sizing
# ═══════════════════════════════════════════════════════════════════

def run_logistic_backtest(df, prob_col, target, pcol,
                          date_col="announcement_date",
                          prob_threshold=0.55, kelly_fraction=0.5,
                          long_only=True, initial_capital=1_000_000,
                          cost_bps=10, max_weight=0.15):
    """
    Backtest the logistic signal using probability-based trade selection
    and Kelly-criterion position sizing.

    Trade rules:
      LONG: P(up) > prob_threshold
      SHORT: P(up) < (1 - prob_threshold)  [only if long_only=False]

    Sizing (half-Kelly):
      edge = |P - 0.5|
      raw_weight ∝ edge
      weight = min(raw_weight * kelly_fraction, max_weight)

    This naturally concentrates capital in high-conviction picks and
    refuses to trade when the model sees no edge.

    Parameters
    ----------
    df : DataFrame with prob_col, target, period and date columns
    prob_col : column name with predicted probabilities
    target : target column name (continuous return)
    pcol : period column name
    date_col : date column name
    prob_threshold : minimum probability to trigger a LONG trade
    kelly_fraction : fraction of Kelly to use (0.5 = half-Kelly)
    long_only : if True, skip SHORT trades
    initial_capital : starting capital
    cost_bps : transaction cost in basis points
    max_weight : maximum per-name weight

    Returns
    -------
    Same as run_backtest_generic: (trades, daily, quarterly, equity)
    """
    df = df.copy()
    df = df[df[prob_col].notna() & df[target].notna()].copy()

    # ── Direction based on probability threshold ──
    df["_log_dir"] = 0
    df.loc[df[prob_col] > prob_threshold, "_log_dir"] = 1
    if not long_only:
        df.loc[df[prob_col] < (1.0 - prob_threshold), "_log_dir"] = -1

    # ── Conviction = edge = |P - 0.5|, scaled to [0, 1] ──
    df["_log_conv"] = (df[prob_col] - 0.5).abs() * 2  # maps [0, 0.5] → [0, 1]

    # ── Kelly sizing ──
    # For each trade date, allocate weight proportional to edge
    df["_log_kelly_edge"] = df["_log_conv"] * kelly_fraction

    # Compute per-date weights
    df = df.sort_values([pcol, date_col, "ticker"])
    trade_mask = df["_log_dir"] != 0
    cost_frac = cost_bps / 10000
    trade_dates = sorted(df.loc[trade_mask, date_col].unique())

    all_trades = []
    capital = initial_capital
    label = f"logistic_p{int(prob_threshold*100)}_k{int(kelly_fraction*100)}"
    if long_only:
        label += "_long"

    for tdate in trade_dates:
        batch = df[(df[date_col] == tdate) & (df["_log_dir"] != 0)]
        if len(batch) < 1:
            continue

        quarter = batch[pcol].iloc[0]
        dirs = batch["_log_dir"].values.astype(int)
        probs = batch[prob_col].values
        edges = batch["_log_kelly_edge"].values
        rets = batch[target].values
        tickers = batch["ticker"].values
        n = len(batch)

        # Weight proportional to edge, capped at max_weight
        total_edge = edges.sum()
        if total_edge > 1e-12:
            weights = edges / total_edge
        else:
            weights = np.ones(n) / n

        weights = np.minimum(weights, max_weight)
        # Do NOT renormalize — invest less when few picks

        # PnL
        gross_rets = dirs * rets / 100
        net_rets = gross_rets - cost_frac
        pnls = capital * weights * net_rets
        total_pnl = pnls.sum()

        for j in range(n):
            all_trades.append(dict(
                strategy=label, date=tdate, quarter=quarter,
                ticker=tickers[j], direction=int(dirs[j]),
                conviction=round(float(edges[j]), 4),
                predicted_prob=round(float(probs[j]), 4),
                weight=round(float(weights[j]), 6),
                capital_pre=round(capital, 2),
                position_size=round(capital * weights[j], 2),
                return_pct=round(rets[j], 4),
                gross_pnl=round(capital * weights[j] * dirs[j] * rets[j] / 100, 2),
                cost=round(capital * weights[j] * cost_frac, 2),
                net_pnl=round(pnls[j], 2),
                hit=int(dirs[j] * rets[j] > 0),
                _conf_reasons=f"P={probs[j]:.3f}",
            ))

        capital += total_pnl

    df_trades = pd.DataFrame(all_trades)
    if len(df_trades) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Aggregate ──
    df_daily = df_trades.groupby(["strategy", "date", "quarter"]).agg(
        n_trades=("net_pnl", "count"),
        n_long=("direction", lambda x: (x > 0).sum()),
        n_short=("direction", lambda x: (x < 0).sum()),
        gross_pnl=("gross_pnl", "sum"),
        total_cost=("cost", "sum"),
        net_pnl=("net_pnl", "sum"),
        hit_rate=("hit", "mean"),
    ).reset_index()

    df_quarterly = df_trades.groupby(["strategy", "quarter"]).agg(
        n_trades=("net_pnl", "count"),
        n_long=("direction", lambda x: (x > 0).sum()),
        n_short=("direction", lambda x: (x < 0).sum()),
        gross_pnl=("gross_pnl", "sum"),
        total_cost=("cost", "sum"),
        net_pnl=("net_pnl", "sum"),
        hit_rate=("hit", "mean"),
        avg_conviction=("conviction", "mean"),
        avg_prob=("predicted_prob", "mean"),
        best_trade=("net_pnl", "max"),
        worst_trade=("net_pnl", "min"),
    ).reset_index()

    cum_cap = initial_capital
    ret_pcts = []
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
# 6. THRESHOLD SWEEP — find optimal P threshold without overfitting
# ═══════════════════════════════════════════════════════════════════

def sweep_thresholds(df, prob_col, target, pcol, date_col="announcement_date",
                     thresholds=None, long_only=True, cost_bps=10,
                     initial_capital=1_000_000, kelly_fraction=0.5):
    """
    Evaluate multiple probability thresholds and report key metrics
    for each. Helps the researcher pick a threshold based on the
    trade-off between trade frequency and accuracy.

    IMPORTANT: This is an in-sample analysis of threshold sensitivity,
    NOT a way to optimise the threshold. The researcher should pick a
    threshold based on economic reasoning (e.g. "I need >55% hit rate
    to be profitable after costs") rather than maximising backtest PnL.

    Parameters
    ----------
    thresholds : list of P thresholds to test (default: 0.50..0.70)

    Returns
    -------
    DataFrame with one row per threshold showing n_trades, hit_rate,
    avg_return, profit_factor, Sharpe, and final capital.
    """
    if thresholds is None:
        thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]

    rows = []
    for th in thresholds:
        trades, _, quarterly, equity = run_logistic_backtest(
            df, prob_col, target, pcol, date_col,
            prob_threshold=th, kelly_fraction=kelly_fraction,
            long_only=long_only, initial_capital=initial_capital,
            cost_bps=cost_bps
        )
        if len(trades) == 0:
            rows.append({"threshold": th, "n_trades": 0})
            continue

        # Import at runtime to avoid circular dependency
        import importlib
        _pv6 = importlib.import_module("pipeline_v6")
        stats = _pv6.compute_trade_stats(trades, cost_bps)
        dr = _pv6.compute_date_returns(trades)

        o = stats["overall"].iloc[0] if len(stats["overall"]) > 0 else {}
        final = equity["capital"].iloc[-1] if len(equity) > 0 else initial_capital

        sharpe = np.nan
        if len(dr) > 1 and dr["portfolio_return"].std() > 0:
            d = dr["portfolio_return"]
            sharpe = d.mean() / d.std() * np.sqrt(min(len(d), 250))

        rows.append({
            "threshold": th,
            "n_trades": len(trades),
            "n_dates": len(dr),
            "hit_rate": o.get("hit_rate", np.nan),
            "avg_return_net": o.get("avg_return_net", np.nan),
            "profit_factor": o.get("profit_factor", np.nan),
            "win_loss_ratio": o.get("win_loss_ratio", np.nan),
            "sharpe": sharpe,
            "final_capital": final,
            "total_return_pct": (final / initial_capital - 1) * 100,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 7. PLOTTING — logistic-specific diagnostics
# ═══════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

# Inherit style from pipeline
POS, NEG, NEUT = "#2E86AB", "#D7263D", "#888"
PAL = ["#2E86AB", "#D7263D", "#F4A261", "#2A9D8F", "#7B2D8E",
       "#E76F51", "#264653", "#E9C46A"]

def _show(fig):
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_calibration(cal_table, label=""):
    """Reliability diagram: predicted probability vs observed frequency."""
    if len(cal_table) == 0:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax1.bar(cal_table["bin_center"], cal_table["observed_freq"],
            width=0.08, color=POS, alpha=0.6, label="Observed")
    ax1.scatter(cal_table["mean_predicted"], cal_table["observed_freq"],
                c=NEG, s=30, zorder=5, label="Mean predicted")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title(f"Calibration (reliability diagram){' — ' + label if label else ''}")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Right: histogram of prediction counts per bin
    ax2.bar(cal_table["bin_center"], cal_table["count"],
            width=0.08, color=PAL[3], alpha=0.6)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction distribution")
    ax2.grid(alpha=0.3, axis="y")

    _show(fig)


def plot_oos_diagnostics(diagnostics, label=""):
    """Per-period OOS metrics: AUC, Brier, logloss, hit rate at P>0.55."""
    if not diagnostics:
        return
    df_d = pd.DataFrame(diagnostics)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) AUC over time
    ax = axes[0, 0]
    vals = df_d["oos_auc"].dropna()
    if len(vals) > 0:
        ax.bar(range(len(vals)), vals.values,
               color=[POS if v > 0.5 else NEG for v in vals.values], alpha=0.6)
        ax.axhline(0.5, c="black", lw=0.5, ls="--", label="Random (0.5)")
        if len(vals) > 3:
            ax.plot(range(len(vals)),
                    vals.rolling(4, min_periods=2).mean().values,
                    c="black", lw=1.2, label="4Q rolling mean")
        ax.set_title(f"OOS AUC  (mean={vals.mean():.3f})")
        ax.set_ylabel("AUC")
        ax.legend(fontsize=7)
    ax.grid(axis="y")

    # (0,1) Brier score over time
    ax = axes[0, 1]
    vals = df_d["oos_brier"].dropna()
    if len(vals) > 0:
        ax.bar(range(len(vals)), vals.values, color=PAL[3], alpha=0.6)
        ax.axhline(0.25, c="black", lw=0.5, ls="--", label="Null (base rate 0.5)")
        ax.set_title(f"OOS Brier Score  (mean={vals.mean():.3f}, lower=better)")
        ax.set_ylabel("Brier score")
        ax.legend(fontsize=7)
    ax.grid(axis="y")

    # (1,0) Hit rate at P>0.55 over time
    ax = axes[1, 0]
    vals = df_d["hit_at_55"].dropna()
    if len(vals) > 0:
        ax.bar(range(len(vals)), vals.values * 100,
               color=[POS if v > 0.5 else NEG for v in vals.values], alpha=0.6)
        base_rates = df_d["test_base_rate"].dropna()
        if len(base_rates) == len(vals):
            ax.plot(range(len(vals)), base_rates.values * 100,
                    c="black", lw=1, ls=":", label="Base rate")
        ax.axhline(50, c="black", lw=0.5, ls="--")
        ax.set_title(f"Hit rate where P>0.55  (mean={vals.mean():.1%})")
        ax.set_ylabel("Hit rate (%)")
        ax.legend(fontsize=7)
    ax.grid(axis="y")

    # (1,1) Number of trades and active features
    ax = axes[1, 1]
    n_above = df_d["n_above_55"]
    n_feat = df_d["n_active_features"]
    x = range(len(df_d))
    ax.bar(x, n_above, color=POS, alpha=0.6, label="Trades (P>0.55)")
    ax2 = ax.twinx()
    ax2.plot(x, n_feat, c=NEG, lw=1.5, marker="o", ms=3, label="Active features")
    ax.set_title("Trade count & model complexity per period")
    ax.set_ylabel("# trades")
    ax2.set_ylabel("# active features (L1)")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y")

    periods = df_d["period"].values
    for a in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        a.set_xticks(range(len(periods)))
        a.set_xticklabels(periods, rotation=45, fontsize=6)

    fig.suptitle(f"Logistic Regression OOS Diagnostics{' — ' + label if label else ''}",
                 fontsize=12, y=1.01)
    _show(fig)


def plot_threshold_sweep(sweep_df):
    """Visualise the threshold sweep results."""
    if len(sweep_df) == 0:
        return
    df = sweep_df[sweep_df["n_trades"] > 0].copy()
    if len(df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (0,0) Hit rate vs threshold
    ax = axes[0, 0]
    ax.plot(df["threshold"], df["hit_rate"] * 100, "o-", color=POS, lw=1.5)
    ax.axhline(50, c="black", lw=0.5, ls="--")
    ax.set_xlabel("P threshold")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Hit rate vs threshold")
    ax.grid(alpha=0.3)

    # (0,1) Trade count vs threshold
    ax = axes[0, 1]
    ax.bar(df["threshold"], df["n_trades"], width=0.015, color=PAL[3], alpha=0.7)
    ax.set_xlabel("P threshold")
    ax.set_ylabel("# trades")
    ax.set_title("Trade count vs threshold")
    ax.grid(axis="y", alpha=0.3)

    # (1,0) Profit factor vs threshold
    ax = axes[1, 0]
    pf = df["profit_factor"].fillna(0)
    ax.plot(df["threshold"], pf, "s-", color=NEG, lw=1.5)
    ax.axhline(1.0, c="black", lw=0.5, ls="--", label="Breakeven")
    ax.set_xlabel("P threshold")
    ax.set_ylabel("Profit factor")
    ax.set_title("Profit factor vs threshold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (1,1) Sharpe vs threshold
    ax = axes[1, 1]
    sh = df["sharpe"].fillna(0)
    ax.plot(df["threshold"], sh, "D-", color=PAL[4], lw=1.5)
    ax.axhline(0, c="black", lw=0.5, ls="--")
    ax.set_xlabel("P threshold")
    ax.set_ylabel("Date-level Sharpe")
    ax.set_title("Sharpe vs threshold")
    ax.grid(alpha=0.3)

    fig.suptitle("Probability Threshold Sensitivity Analysis", fontsize=12, y=1.01)
    _show(fig)


def plot_feature_importance(diagnostics, features, top_n=15):
    """Average absolute coefficient across all OOS periods (L1 path)."""
    if not diagnostics:
        return

    # Collect all coefficient dictionaries
    all_coefs = {}
    n_periods = 0
    for d in diagnostics:
        for f, c in d.get("coefs", {}).items():
            if f not in all_coefs:
                all_coefs[f] = []
            all_coefs[f].append(abs(c))
        n_periods += 1

    if not all_coefs:
        return

    # Average |coef| and selection frequency
    rows = []
    for f, vals in all_coefs.items():
        rows.append({
            "feature": f,
            "avg_abs_coef": np.mean(vals),
            "selection_freq": len(vals) / n_periods,  # fraction of periods where L1 kept it
        })
    df_imp = pd.DataFrame(rows).sort_values("avg_abs_coef", ascending=False).head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(df_imp) * 0.35)))

    y = np.arange(len(df_imp))
    feats = df_imp["feature"].values

    # Left: avg |coefficient|
    ax1.barh(y, df_imp["avg_abs_coef"], color=POS, alpha=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(feats, fontsize=8)
    ax1.set_xlabel("Average |coefficient|")
    ax1.set_title("Feature importance (L1 coefficients)")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    # Right: selection frequency (how often L1 kept it nonzero)
    ax2.barh(y, df_imp["selection_freq"] * 100, color=PAL[3], alpha=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(feats, fontsize=8)
    ax2.set_xlabel("Selection frequency (%)")
    ax2.set_title("L1 selection stability")
    ax2.set_xlim(0, 105)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature Importance & Stability (OOS)", fontsize=12)
    _show(fig)


def plot_logistic_summary_table(validation, diagnostics, sweep_df=None, label=""):
    """Single-page summary table of all statistical tests."""
    rows = []

    # Brier decomposition
    b = validation.get("brier", {})
    rows.append(["Brier Score", f'{b.get("brier_score", np.nan):.4f}'])
    rows.append(["  Reliability (calibration error)", f'{b.get("reliability", np.nan):.4f}'])
    rows.append(["  Resolution (discrimination)", f'{b.get("resolution", np.nan):.4f}'])
    rows.append(["  Skill Score (vs null)", f'{b.get("skill_score", np.nan):.3f}'])

    # AUC
    a = validation.get("auc", {})
    rows.append(["AUC", f'{a.get("auc", np.nan):.3f}  [{a.get("ci_95_lower", np.nan):.3f}, {a.get("ci_95_upper", np.nan):.3f}]'])

    # Hosmer-Lemeshow
    hl = validation.get("hosmer_lemeshow", {})
    rows.append(["Hosmer-Lemeshow p-value", f'{hl.get("p_value", np.nan):.3f}  '
                 f'({"well-calibrated" if hl.get("well_calibrated") else "miscalibrated"})'])

    # Log-loss
    ll = validation.get("logloss", {})
    rows.append(["Log-loss (model)", f'{ll.get("model", np.nan):.4f}'])
    rows.append(["Log-loss (null)", f'{ll.get("null_model", np.nan):.4f}'])
    rows.append(["Log-loss improvement", f'{ll.get("improvement_pct", 0):.1f}%'])

    # Permutation test
    pt = validation.get("permutation_test", {})
    rows.append(["Hit rate (traded)", f'{pt.get("observed_hit_rate", np.nan):.3f}'])
    rows.append(["Permutation p-value", f'{pt.get("p_value", np.nan):.3f}  '
                 f'({"significant" if pt.get("significant_at_05") else "not significant"})'])

    # OOS summary from diagnostics
    if diagnostics:
        df_d = pd.DataFrame(diagnostics)
        rows.append(["", ""])
        rows.append(["── OOS Period Averages ──", ""])
        rows.append(["Mean OOS AUC", f'{df_d["oos_auc"].mean():.3f}'])
        rows.append(["Mean OOS Brier", f'{df_d["oos_brier"].mean():.4f}'])
        rows.append(["Mean active features", f'{df_d["n_active_features"].mean():.1f}'])
        rows.append(["Median C selected", f'{df_d["best_C"].median():.4f}'])
        rows.append(["Periods with AUC > 0.5", f'{(df_d["oos_auc"] > 0.5).sum()}/{len(df_d)}'])

    fig, ax = plt.subplots(figsize=(7, len(rows) * 0.33 + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   cellLoc="left", loc="center", colWidths=[0.55, 0.45])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E0E0E0")
        if c == 0:
            cell.set_text_props(fontweight="normal")
    ax.set_title(f"Logistic Signal Statistical Validation{' — ' + label if label else ''}",
                 fontsize=12, pad=20)
    _show(fig)