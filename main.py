import json
from pathlib import Path
from dataclasses import dataclass
from typing import List
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt



def apply_root_style():
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.size": 12,

        # Axes
        "axes.linewidth": 1.2,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.spines.top": True,
        "axes.spines.right": True,

        # Major ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,

        # Minor ticks  👈
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,

        # Ticks on all sides
        "xtick.top": True,
        "ytick.right": True,

        # Figure
        "figure.figsize": (7, 5),
        "figure.dpi": 100,

        # No grid (ROOT default)
        "axes.grid": False,
    })

# -----------------------
# Domain model
# -----------------------

@dataclass
class Report:
    report_id: str
    total_cost: float
    documents_total: int


def get_team_costs(reports_dir: Path, team_name: str) -> list[Report]:
    """
    Load JSON files and extract the cost for a specific team
    """
    reports: list[Report] = []
    for path in reports_dir.glob("*.json"):
        with path.open() as f:
            data = json.load(f)
        team_cost = data["cost"]["by_team"].get(team_name, {})
        total = sum(team_cost.values()) if team_cost else 0.0
        reports.append(
            Report(
                report_id=data["report_id"],
                total_cost=total,
                documents_total=data["usage"]["components"]["documents"]["total"],
            )
        )
    return reports

def fit_straight_line(
    reports: list[Report],
) -> tuple[Callable[[float], float], str]:
    """
    Fit y = m x + c where:
      x = total documents
      y = total cost (USD)

    Returns:
      - callable f(x)
      - equation string
    """
    x = np.array([r.documents_total for r in reports], dtype=float)
    y = np.array([r.total_cost for r in reports], dtype=float)

    # Least-squares linear fit
    m, c = np.polyfit(x, y, deg=1)

    def f(x_val: float) -> float:
        return m * x_val + c

    equation = f"Cost = ({m:.4f}) × Documents + ({c:.4f}) USD"

    return f, equation


# -----------------------
# Load reports
# -----------------------

def load_reports(directory: Path) -> List[Report]:
    reports: List[Report] = []

    for file_path in directory.glob("*.json"):
        with file_path.open() as f:
            data = json.load(f)

        reports.append(
            Report(
                report_id=data["report_id"],
                total_cost=data["cost"]["total"],
                documents_total=data["usage"]["components"]["documents"]["total"],
            )
        )

    return reports

def fit_straight_line_with_uncertainty(
    reports: list[Report],
) -> tuple[
    Callable[[float], float],
    Callable[[float], float],
    float,
    float,
    float,
    float,
]:
    """
    Fits y = m x + c with uncertainties.

    Returns:
      f(x)            -> fitted line
      sigma_y(x)      -> 1σ uncertainty at x
      m               slope
      sigma_m         slope uncertainty
      c               intercept
      sigma_c         intercept uncertainty
    """
    x = np.array([r.documents_total for r in reports], dtype=float)
    y = np.array([r.total_cost for r in reports], dtype=float)

    n = len(x)
    x_mean = np.mean(x)

    # Fit
    m, c = np.polyfit(x, y, deg=1)

    # Residuals
    y_fit = m * x + c
    residuals = y - y_fit

    # Variance estimate
    s2 = np.sum(residuals**2) / (n - 2)

    Sxx = np.sum((x - x_mean) ** 2)

    # Parameter uncertainties
    sigma_m = np.sqrt(s2 / Sxx)
    sigma_c = np.sqrt(s2 * (1.0 / n + x_mean**2 / Sxx))

    def f(x_val: float) -> float:
        return m * x_val + c

    def sigma_y(x_val: float) -> float:
        return np.sqrt(
            s2 * (1.0 / n + (x_val - x_mean) ** 2 / Sxx)
        )

    return f, sigma_y, m, sigma_m, c, sigma_c

# -----------------------
# Plotting
# -----------------------
def plot_cost_vs_documents(reports: list[Report]) -> None:
    apply_root_style()

    x = np.array([r.documents_total for r in reports], dtype=float)
    y = np.array([r.total_cost for r in reports], dtype=float)

    fig, ax = plt.subplots()

    # Data
    ax.scatter(
        x,
        y,
        marker="x",
        s=30,
        color="red",
        linewidths=1.2,
        label="Data",
    )

    # Fit + uncertainty
    fit_fn, sigma_y, m, sigma_m, c, sigma_c = (
        fit_straight_line_with_uncertainty(reports)
    )

    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = fit_fn(x_fit)
    y_err = sigma_y(x_fit)

    # Fit line
    ax.plot(x_fit, y_fit, linewidth=1.5, label="Linear fit")

    # 1σ uncertainty band
    ax.fill_between(
        x_fit,
        y_fit - y_err,
        y_fit + y_err,
        alpha=0.25,
        label="1σ uncertainty",
    )

    # Labels
    ax.set_xlabel("Total Documents")
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Total Cost vs Document Usage")

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.margins(0.05)

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # ROOT-style fit printout
    print(
        f"Fit result:\n"
        f"  Cost = m × Documents + c\n"
        f"  m = {m:.5f} ± {sigma_m:.5f} USD/doc\n"
        f"  c = {c:.5f} ± {sigma_c:.5f} USD"
    )

def plot_and_save(reports: list[Report], title: str, filename: str):
    apply_root_style()

    x = np.array([r.documents_total for r in reports], dtype=float)
    y = np.array([r.total_cost for r in reports], dtype=float)

    fig, ax = plt.subplots()

    # Data points
    ax.scatter(
        x,
        y,
        marker="x",
        s=30,
        color="red",
        linewidths=1.2,
        label="Data",
    )

    # Fit + uncertainty
    fit_fn, sigma_y, m, sigma_m, c, sigma_c = fit_straight_line_with_uncertainty(reports)
    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = fit_fn(x_fit)
    y_err = sigma_y(x_fit)

    # Fit line
    ax.plot(x_fit, y_fit, linewidth=1.5, label="Linear fit")

    # 1σ uncertainty band
    ax.fill_between(x_fit, y_fit - y_err, y_fit + y_err, alpha=0.25, label="1σ uncertainty")

    # Labels
    ax.set_xlabel("Total Documents")
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title(title)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.margins(0.05)

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

    print(f"Saved plot to {filename}")
    print(f"Fit result for {title}:\n"
          f"  Cost = m × Documents + c\n"
          f"  m = {m:.5f} ± {sigma_m:.5f} USD/doc\n"
          f"  c = {c:.5f} ± {sigma_c:.5f} USD")
    
def report_id_docs(reports: list[Report]):
    """
    Prints each report_id along with the number of documents used.
    """
    for r in sorted(reports, key=lambda x: x.documents_total, reverse=True):
        print(f"{r.report_id}: {r.documents_total} docs")

def plot_all_teams_on_same_graph(reports_dir: Path, filename: str = "all_teams.png"):
    apply_root_style()

    teams = ["total", "entity_resolution", "investigative_flow", "report_content_generation", "other"]
    colors = ["black", "blue", "green", "orange", "purple"]
    labels = ["Total", "Entity Resolution", "Investigative Flow", "Report Content Generation", "Other"]

    fig, ax = plt.subplots()

    for team, color, label in zip(teams, colors, labels):
        # Load appropriate reports
        if team == "total":
            reports = load_reports(reports_dir)
        else:
            reports = get_team_costs(reports_dir, team)

        x = np.array([r.documents_total for r in reports], dtype=float)
        y = np.array([r.total_cost for r in reports], dtype=float)

        # Scatter points
        ax.scatter(x, y, marker="x", s=30, color=color, linewidths=1.2, label=f"{label} Data")

        # Fit + uncertainty
        fit_fn, sigma_y, m, sigma_m, c, sigma_c = fit_straight_line_with_uncertainty(reports)
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = fit_fn(x_fit)
        y_err = sigma_y(x_fit)

        # Fit line
        ax.plot(x_fit, y_fit, color=color, linewidth=1.5, label=f"{label} Fit")

        # 1σ uncertainty band
        ax.fill_between(x_fit, y_fit - y_err, y_fit + y_err, color=color, alpha=0.15)

    # Labels
    ax.set_xlabel("Total Documents")
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Cost vs Documents per Team")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.margins(0.05)

    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved combined team plot to {filename}")

def print_averages(reports: list[Report]) -> None:
    """
    Prints the average document count and average cost.
    """
    if not reports:
        print("No reports provided.")
        return

    avg_docs = np.mean([r.documents_total for r in reports])
    avg_cost = np.mean([r.total_cost for r in reports])

    print(
        f"Average documents: {avg_docs:.2f}\n"
        f"Average cost: ${avg_cost:.2f}"
    )

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    reports_dir = Path("report_costs_3")
    print_averages(load_reports(reports_dir))
    report_id_docs(load_reports(reports_dir))
    # Total cost plot
    total_reports = load_reports(reports_dir)
    plot_and_save(total_reports, "Total Cost vs Documents", "total.png")
    plot_all_teams_on_same_graph(reports_dir, "all_teams.png")
    # Per-team plots
    teams = ["entity_resolution", "investigative_flow", "report_content_generation", "other"]
    for team in teams:
        team_reports = get_team_costs(reports_dir, team)
        plot_and_save(team_reports, f"{team.replace('_', ' ').title()} Cost vs Documents",
                      f"{team}.png")
        
    