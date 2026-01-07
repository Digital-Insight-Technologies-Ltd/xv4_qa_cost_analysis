import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

# Use ROOT-like style
hep.style.use("ROOT")

# Load the CSV
df = pd.read_csv("xdoc-counts.csv")

# Basic sanity check (optional)
print(df.head())
print(df["xdoc_count"].describe())

# Plot histogram of xdoc_count
plt.figure()
plt.hist(
    df["xdoc_count"],
    bins=60,          # e.g. 50 units per bin
    range=(0, 3000),
    histtype="step",
    linewidth=1.5
)

plt.xlabel("xdoc_count")
plt.ylabel("Frequency")
plt.title("Histogram of xdoc_count")
plt.show()


plt.xlim(0, 3000)
