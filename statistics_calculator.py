"""
Statistics Calculator
======================
Original implementation by Protoy Chowdhury
github.com/Protoy-git

Covers core statistical concepts using Python and NumPy:
- Descriptive statistics
- Probability distributions
- Hypothesis testing (t-test)
- Correlation and covariance
- Data summary reporting
"""

import numpy as np


# ─── DESCRIPTIVE STATISTICS ───────────────────────────────────

def descriptive_stats(data):
    """
    Computes full descriptive statistics for a dataset.
    Returns a dictionary of statistical measures.
    """
    data = np.array(data, dtype=float)
    n = len(data)

    stats = {
        "count":        n,
        "mean":         np.mean(data),
        "median":       np.median(data),
        "std_dev":      np.std(data, ddof=1),        # sample std dev
        "variance":     np.var(data, ddof=1),         # sample variance
        "min":          np.min(data),
        "max":          np.max(data),
        "range":        np.max(data) - np.min(data),
        "q1":           np.percentile(data, 25),
        "q3":           np.percentile(data, 75),
        "iqr":          np.percentile(data, 75) - np.percentile(data, 25),
        "skewness":     _skewness(data),
        "kurtosis":     _kurtosis(data),
    }
    return stats


def _skewness(data):
    """Pearson's skewness — measures asymmetry of distribution."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)


def _kurtosis(data):
    """Excess kurtosis — measures tail heaviness relative to normal distribution."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * \
           np.sum(((data - mean) / std) ** 4) - \
           (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def print_descriptive(data, label="Dataset"):
    """Prints a formatted descriptive statistics report."""
    stats = descriptive_stats(data)
    print(f"\n{'=' * 45}")
    print(f"  DESCRIPTIVE STATISTICS — {label}")
    print(f"{'=' * 45}")
    for key, val in stats.items():
        print(f"  {key:<15}: {round(val, 4)}")
    print(f"{'=' * 45}")


# ─── PROBABILITY DISTRIBUTIONS ────────────────────────────────

def normal_distribution(mu, sigma, x_values):
    """
    Normal (Gaussian) distribution PDF.
    f(x) = (1 / sigma*sqrt(2pi)) * exp(-0.5 * ((x - mu)/sigma)^2)
    """
    x = np.array(x_values)
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coefficient * exponent


def binomial_probability(n, k, p):
    """
    Binomial probability — P(X = k) for n trials with success prob p.
    Uses the binomial coefficient formula.
    """
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def poisson_probability(lam, k):
    """
    Poisson probability — P(X = k) given average rate lambda.
    P(X=k) = (lambda^k * e^-lambda) / k!
    """
    from math import factorial, exp
    return (lam ** k * exp(-lam)) / factorial(k)


# ─── HYPOTHESIS TESTING ───────────────────────────────────────

def one_sample_t_test(data, mu_0, alpha=0.05):
    """
    One-sample t-test.
    Tests whether the sample mean differs significantly from mu_0.
    H0: mean == mu_0
    H1: mean != mu_0
    """
    data = np.array(data, dtype=float)
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    t_stat = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
    df = n - 1

    # Critical value approximation (two-tailed, common alpha levels)
    critical_values = {0.10: 1.645, 0.05: 1.960, 0.01: 2.576}
    critical = critical_values.get(alpha, 1.960)

    reject = abs(t_stat) > critical

    print(f"\n{'=' * 45}")
    print(f"  ONE-SAMPLE T-TEST")
    print(f"{'=' * 45}")
    print(f"  Sample size   : {n}")
    print(f"  Sample mean   : {round(sample_mean, 4)}")
    print(f"  Hypothesized  : {mu_0}")
    print(f"  Std deviation : {round(sample_std, 4)}")
    print(f"  t-statistic   : {round(t_stat, 4)}")
    print(f"  Degrees of f  : {df}")
    print(f"  Alpha level   : {alpha}")
    print(f"  Critical val  : ±{critical}")
    print(f"  Decision      : {'REJECT H0 — significant difference' if reject else 'FAIL TO REJECT H0 — no significant difference'}")
    print(f"{'=' * 45}")
    return t_stat, reject


# ─── CORRELATION & COVARIANCE ─────────────────────────────────

def correlation_analysis(x, y):
    """
    Computes Pearson correlation coefficient and covariance.
    r = cov(X,Y) / (std_X * std_Y)
    Range: -1 (perfect negative) to +1 (perfect positive)
    """
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    cov = np.cov(x, y, ddof=1)[0][1]
    r = np.corrcoef(x, y)[0][1]

    if abs(r) >= 0.8:
        strength = "Strong"
    elif abs(r) >= 0.5:
        strength = "Moderate"
    else:
        strength = "Weak"
    direction = "positive" if r > 0 else "negative"

    print(f"\n{'=' * 45}")
    print(f"  CORRELATION ANALYSIS")
    print(f"{'=' * 45}")
    print(f"  Covariance    : {round(cov, 4)}")
    print(f"  Pearson r     : {round(r, 4)}")
    print(f"  Interpretation: {strength} {direction} correlation")
    print(f"{'=' * 45}")
    return r, cov


# ─── DEMO ─────────────────────────────────────────────────────

def demo():
    print("\n" + "=" * 45)
    print("  STATISTICS CALCULATOR DEMO")
    print("  Protoy Chowdhury — github.com/Protoy-git")
    print("=" * 45)

    # Sample dataset — exam scores
    scores = [72, 85, 90, 68, 77, 95, 88, 73, 81, 92,
              65, 78, 84, 91, 70, 88, 76, 82, 69, 94]

    print_descriptive(scores, "Exam Scores")

    # Hypothesis test — is mean significantly different from 75?
    one_sample_t_test(scores, mu_0=75, alpha=0.05)

    # Correlation between study hours and scores
    study_hours = [3, 5, 6, 2, 4, 7, 6, 3, 5, 7,
                   2, 4, 5, 7, 3, 6, 4, 5, 2, 7]
    correlation_analysis(study_hours, scores)

    # Probability examples
    print(f"\n{'=' * 45}")
    print(f"  PROBABILITY EXAMPLES")
    print(f"{'=' * 45}")
    p_binom = binomial_probability(n=10, k=3, p=0.4)
    print(f"  Binomial P(X=3 | n=10, p=0.4) : {round(p_binom, 4)}")
    p_poisson = poisson_probability(lam=5, k=3)
    print(f"  Poisson  P(X=3 | lambda=5)    : {round(p_poisson, 4)}")
    print(f"{'=' * 45}")


if __name__ == "__main__":
    demo()
