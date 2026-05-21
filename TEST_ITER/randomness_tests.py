import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chisquare, kstest, entropy
from statsmodels.sandbox.stats.runs import runstest_1samp

# -------------------------------
# 1. LOAD DATA
# -------------------------------
file_path  = "C:/Users/ojask/Documents/DSL/Project/random_values.csv"
column_name = "DecimalValue"   # <-- change this

df = pd.read_csv(file_path)

# Drop NaNs just in case
data = df[column_name].dropna().values

# -------------------------------
# 2. SAMPLE DATA (adjust if needed)
# -------------------------------
sample_size = 10000
if len(data) > sample_size:
    np.random.seed(42)
    data = np.random.choice(data, size=sample_size, replace=False)

print(f"Sample size used: {len(data)}")

# Normalize data to [0,1] for KS test
data_min, data_max = np.min(data), np.max(data)
data_norm = (data - data_min) / (data_max - data_min)

# -------------------------------
# 3. CHI-SQUARE TEST (Uniformity)
# -------------------------------
bins = 50
counts, _ = np.histogram(data, bins=bins)

expected = np.ones_like(counts) * np.mean(counts)

chi2_stat, chi2_p = chisquare(counts, expected)

print("\nChi-square test:")
print(f"Statistic = {chi2_stat:.4f}, p-value = {chi2_p:.4f}")

# -------------------------------
# 4. KS TEST (Uniform distribution)
# -------------------------------
ks_stat, ks_p = kstest(data_norm, 'uniform')

print("\nKS test:")
print(f"Statistic = {ks_stat:.4f}, p-value = {ks_p:.4f}")

# -------------------------------
# 5. RUNS TEST
# -------------------------------
z_stat, runs_p = runstest_1samp(data)

print("\nRuns test:")
print(f"Z = {z_stat:.4f}, p-value = {runs_p:.4f}")

# -------------------------------
# 6. AUTOCORRELATION (lag = 1)
# -------------------------------
autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]

print("\nAutocorrelation (lag 1):")
print(f"{autocorr:.6f}")

# -------------------------------
# 7. ENTROPY
# -------------------------------
probabilities = counts / np.sum(counts)
ent = entropy(probabilities)

print("\nEntropy:")
print(f"{ent:.4f}")

# -------------------------------
# 8. HISTOGRAM
# -------------------------------
plt.hist(data, bins=50)
plt.title("Histogram of Sample Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 9. SIMPLE INTERPRETATION
# -------------------------------
print("\n--- Interpretation Guide ---")

if chi2_p > 0.05:
    print("Chi-square: PASS (looks uniform)")
else:
    print("Chi-square: FAIL (non-uniform)")

if ks_p > 0.05:
    print("KS test: PASS (matches uniform distribution)")
else:
    print("KS test: FAIL")

if runs_p > 0.05:
    print("Runs test: PASS (sequence looks random)")
else:
    print("Runs test: FAIL")

if abs(autocorr) < 0.05:
    print("Autocorrelation: PASS (no strong correlation)")
else:
    print("Autocorrelation: WARNING (possible pattern)")

print("Entropy: Higher is better (compare across tests)")