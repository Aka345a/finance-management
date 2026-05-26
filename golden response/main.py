import datetime
import json
import logging
import os
from decimal import Decimal

import numpy as np
import pandas as pd

# Django bootstrap
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=False,
        SECRET_KEY="golden-response-secret-key-change-in-prod",
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "finance",
        ],
        TIME_ZONE="UTC",
        USE_TZ=True,
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )

django.setup()

# ML / metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finance.system")


# ============================================================
# SECTION 2 - DJANGO MODELS
# ============================================================

from django.contrib.auth.models import Group, Permission, User
from django.contrib.contenttypes.models import ContentType
from django.core.validators import MinValueValidator
from django.db import models


class Expense(models.Model):
    """Records a single financial transaction for a user."""

    CATEGORY_CHOICES = [
        ("Food", "Food & Dining"),
        ("Utilities", "Bills & Utilities"),
        ("Entertainment", "Entertainment"),
        ("Transport", "Transportation"),
        ("Housing", "Housing"),
        ("Healthcare", "Medical & Healthcare"),
        ("Shopping", "Shopping"),
        ("Misc", "Miscellaneous"),
    ]

    PAYMENT_CHOICES = [
        ("UPI", "UPI"),
        ("Card", "Debit / Credit Card"),
        ("Cash", "Cash"),
        ("NetBanking", "Net Banking"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="expenses")
    expense_date = models.DateField(db_index=True)
    amount = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        validators=[MinValueValidator(Decimal("0.01"))],
    )
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, db_index=True)
    description = models.TextField(blank=True, default="")
    payment_method = models.CharField(
        max_length=50, choices=PAYMENT_CHOICES, default="UPI"
    )
    currency = models.CharField(max_length=3, default="INR")

    class Meta:
        app_label = "finance"
        ordering = ["-expense_date"]

    def __str__(self) -> str:
        return f"{self.category} INR {self.amount} on {self.expense_date}"


class BudgetConfiguration(models.Model):
    """Monthly budget limit per user and category."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="budgets")
    month = models.DateField(db_index=True)
    category = models.CharField(max_length=50, choices=Expense.CATEGORY_CHOICES)
    budget_limit = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        validators=[MinValueValidator(Decimal("0.00"))],
    )

    class Meta:
        app_label = "finance"
        unique_together = ("user", "month", "category")

    def __str__(self) -> str:
        return f"User {self.user_id} | {self.month:%B %Y} | {self.category}"


# ============================================================
# SECTION 3 - ETL & FEATURE ENGINEERING PIPELINE
# ============================================================


class FinancialDataPipeline:
    """
    Transforms raw ORM querysets into an ML-ready feature matrix.

    Steps:
      1. Type coercion & date parsing
      2. Monthly aggregation per user
      3. Z-score anomaly detection
      4. Budget boundary alignment
      5. Lag feature engineering (previous month spending)
      6. Target variable derivation (savings = budget - spending)
    """

    def clean(
        self,
        expenses_df: pd.DataFrame,
        budgets_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Coerce types and derive period columns."""
        if not expenses_df.empty:
            expenses_df["amount"] = pd.to_numeric(
                expenses_df["amount"], errors="coerce"
            ).fillna(0.0)
            expenses_df["expense_date"] = pd.to_datetime(expenses_df["expense_date"])
            expenses_df["year_month"] = expenses_df["expense_date"].dt.to_period("M")

        if not budgets_df.empty:
            budgets_df["budget_limit"] = pd.to_numeric(
                budgets_df["budget_limit"], errors="coerce"
            ).fillna(0.0)
            budgets_df["year_month"] = pd.to_datetime(budgets_df["month"]).dt.to_period(
                "M"
            )

        return expenses_df, budgets_df

    def build_features(
        self,
        expenses_df: pd.DataFrame,
        budgets_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a feature matrix with monthly spending, budget, anomaly, lag, and
        savings target columns.
        """
        expenses_df, budgets_df = self.clean(expenses_df, budgets_df)

        if expenses_df.empty:
            logger.warning("No expense records found; returning empty feature space.")
            return pd.DataFrame()

        monthly = (
            expenses_df.groupby(["user_id", "year_month"])
            .agg(
                total_spending=("amount", "sum"),
                transaction_count=("amount", "count"),
                avg_daily_spending=("amount", lambda x: x.sum() / 30.0),
            )
            .reset_index()
        )

        user_stats = (
            expenses_df.groupby("user_id")["amount"]
            .agg(mean_spend="mean", std_spend="std")
            .reset_index()
        )
        expenses_df = expenses_df.merge(user_stats, on="user_id", how="left")
        expenses_df["std_spend"] = expenses_df["std_spend"].fillna(1e-5)
        expenses_df["is_anomaly"] = (
            (expenses_df["amount"] - expenses_df["mean_spend"])
            / expenses_df["std_spend"]
        ) > 2.0

        anomaly_counts = (
            expenses_df.groupby(["user_id", "year_month"])["is_anomaly"]
            .sum()
            .reset_index()
            .rename(columns={"is_anomaly": "anomaly_count"})
        )

        features = monthly.merge(anomaly_counts, on=["user_id", "year_month"], how="left")
        features["anomaly_count"] = features["anomaly_count"].fillna(0)

        if not budgets_df.empty:
            budget_monthly = (
                budgets_df.groupby(["user_id", "year_month"])["budget_limit"]
                .sum()
                .reset_index()
            )
            features = features.merge(
                budget_monthly, on=["user_id", "year_month"], how="left"
            )
        else:
            features["budget_limit"] = np.nan

        features["budget_limit"] = features["budget_limit"].fillna(
            features["total_spending"] * 1.2
        )

        features = features.sort_values(["user_id", "year_month"])
        features["prev_month_spending"] = (
            features.groupby("user_id")["total_spending"].shift(1).fillna(0.0)
        )

        features["target_savings"] = (
            features["budget_limit"] - features["total_spending"]
        )

        logger.info("Feature matrix built: %d rows x %d cols", *features.shape)
        return features


# ============================================================
# SECTION 4 - ML MODEL SELECTION ENGINE
# ============================================================


class SavingsPredictorEngine:
    """
    Trains LinearRegression and GradientBoostingRegressor, selects the champion
    model by cross-validated R2 when data is sufficient, or falls back
    gracefully for small datasets.
    """

    FEATURE_COLS = [
        "total_spending",
        "transaction_count",
        "avg_daily_spending",
        "anomaly_count",
        "prev_month_spending",
    ]
    TARGET_COL = "target_savings"

    def __init__(self) -> None:
        self.champion_model = None
        self.champion_metrics: dict = {}

    def fit(self, dataset: pd.DataFrame) -> dict:
        """Train candidate models and select the best one."""
        if dataset.empty or len(dataset) < 2:
            return self._fallback("insufficient data")

        X = dataset[self.FEATURE_COLS].astype(float)
        y = dataset[self.TARGET_COL].astype(float)

        if len(dataset) < 5:
            model = LinearRegression().fit(X, y)
            preds = model.predict(X)
            self.champion_model = model
            self.champion_metrics = self._score(
                "LinearRegression (small-data)", y, preds
            )
            return self.champion_metrics

        candidates = {
            "LinearRegression": LinearRegression(),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
            ),
        }

        best_name, best_model, best_cv = None, None, -np.inf

        cv_folds = 2 if len(dataset) < 6 else min(3, len(dataset))

        for name, model in candidates.items():
            cv_scores = cross_val_score(
                model, X, y, cv=cv_folds, scoring="r2"
            )
            mean_cv = float(cv_scores.mean())
            logger.info("%s cross-val R2: %.4f", name, mean_cv)
            if np.isfinite(mean_cv) and mean_cv > best_cv:
                best_cv, best_name, best_model = mean_cv, name, model

        if best_model is None:
            best_name = "LinearRegression (cv-fallback)"
            best_model = LinearRegression()
            best_cv = 0.0

        best_model.fit(X, y)
        preds = best_model.predict(X)

        self.champion_model = best_model
        self.champion_metrics = {
            **self._score(best_name, y, preds),
            "cv_r2": round(best_cv, 4),
        }

        logger.info("Champion model selected: %s", best_name)
        return self.champion_metrics

    def predict(self, row: pd.DataFrame) -> float:
        """Predict savings for a single feature row."""
        if self.champion_model is None:
            raise RuntimeError("Call fit() before predict().")
        return float(self.champion_model.predict(row[self.FEATURE_COLS].astype(float))[0])

    @staticmethod
    def _score(name: str, y_true, y_pred) -> dict:
        return {
            "algorithm": name,
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }

    def _fallback(self, reason: str) -> dict:
        """Use a deterministic zero-savings predictor when data is absent."""
        logger.warning("Fallback predictor activated: %s", reason)

        class _ZeroPredictor:
            def predict(self, X):
                return np.zeros(len(X))

        self.champion_model = _ZeroPredictor()
        self.champion_metrics = {
            "algorithm": f"ZeroPredictor ({reason})",
            "mae": 0.0,
            "mse": 0.0,
            "r2": 0.0,
        }
        return self.champion_metrics


# ============================================================
# SECTION 5 - FINANCIAL PROFILE GENERATOR
# ============================================================


def generate_financial_profile(user: User) -> dict:
    """Run the full pipeline for one user and return a structured report."""
    expenses_qs = Expense.objects.filter(user=user).values()
    budgets_qs = BudgetConfiguration.objects.filter(user=user).values()

    expenses_df = pd.DataFrame(list(expenses_qs))
    budgets_df = pd.DataFrame(list(budgets_qs))

    if expenses_df.empty:
        raise ValueError(f"No expense records found for user {user.username!r}.")

    pipeline = FinancialDataPipeline()
    feature_space = pipeline.build_features(expenses_df, budgets_df)

    engine = SavingsPredictorEngine()
    metrics = engine.fit(feature_space)

    latest = feature_space.tail(1)
    predicted_savings = engine.predict(latest)

    current_spending = float(latest["total_spending"].iloc[0])
    budget_limit = float(latest["budget_limit"].iloc[0])

    overspending = current_spending > budget_limit
    overspend_ratio = current_spending / budget_limit if budget_limit > 0 else 1.0
    confidence_score = round(min(max((overspend_ratio - 1.0) * 2.0, 0.0), 1.0), 2)

    alerts = []
    if overspending:
        alerts.append(
            f"Budget exceeded by {overspend_ratio - 1:.1%} "
            f"(confidence: {confidence_score})."
        )
    if predicted_savings < 0:
        alerts.append("Negative savings trajectory predicted for the next month.")
    if feature_space["anomaly_count"].iloc[-1] > 0:
        alerts.append(
            f"{int(feature_space['anomaly_count'].iloc[-1])} anomalous "
            "transaction(s) detected this period."
        )

    chart_labels = [str(p) for p in feature_space["year_month"]]
    chart_spending = feature_space["total_spending"].round(2).tolist()
    chart_budget = feature_space["budget_limit"].round(2).tolist()
    chart_savings = feature_space["target_savings"].round(2).tolist()

    return {
        "user_id": user.id,
        "username": user.username,
        "metrics": {
            "current_monthly_spending": round(current_spending, 2),
            "budget_limit": round(budget_limit, 2),
            "predicted_savings_next_month": round(predicted_savings, 2),
            "overspending_flagged": overspending,
            "overspending_confidence": confidence_score,
            "model": metrics,
        },
        "alerts": alerts,
        "chart_data": {
            "labels": chart_labels,
            "spending": chart_spending,
            "budget": chart_budget,
            "savings": chart_savings,
        },
    }


# ============================================================
# SECTION 6 - DASHBOARD HTML GENERATOR
# ============================================================

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Personal Finance Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --accent: #6c63ff;
    --accent2: #f7c948;
    --danger: #ff4f6a;
    --success: #3ecf8e;
    --text: #e2e8f0;
    --muted: #8892a4;
    --radius: 12px;
    --font: 'Segoe UI', system-ui, sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 32px 24px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
    margin-bottom: 32px;
  }
  header h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: 0; }
  header span { font-size: .85rem; color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 28px; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 22px;
  }
  .card-label { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 8px; }
  .card-value { font-size: 1.65rem; font-weight: 700; }
  .card-value.danger { color: var(--danger); }
  .card-value.success { color: var(--success); }
  .card-value.accent { color: var(--accent2); }
  .charts { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 28px; }
  .chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
  .chart-card h3 { font-size: .85rem; color: var(--muted); margin-bottom: 14px; }
  canvas { max-height: 260px; }
  .alerts { display: flex; flex-direction: column; gap: 10px; }
  .alert {
    background: rgba(255,79,106,.08);
    border-left: 3px solid var(--danger);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: .9rem;
    color: #ffb3be;
  }
  .alert.ok {
    background: rgba(62,207,142,.08);
    border-color: var(--success);
    color: #a0f0cf;
  }
  .model-badge {
    display: inline-block;
    background: rgba(108,99,255,.15);
    color: var(--accent);
    border: 1px solid rgba(108,99,255,.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: .78rem;
    margin-top: 6px;
  }
  footer { text-align: center; color: var(--muted); font-size: .78rem; margin-top: 40px; }
  @media (max-width: 700px) {
    body { padding: 20px 14px; }
    header { align-items: flex-start; flex-direction: column; gap: 10px; }
    .charts { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div>
    <h1>Personal Finance Dashboard</h1>
    <p style="color:var(--muted);margin-top:4px;font-size:.9rem;">AI-powered savings & overspending intelligence</p>
  </div>
  <span id="ts"></span>
</header>

<div class="grid">
  <div class="card">
    <div class="card-label">Current Spending</div>
    <div class="card-value danger">INR __CUR_SPEND__</div>
  </div>
  <div class="card">
    <div class="card-label">Budget Limit</div>
    <div class="card-value">INR __BUDGET__</div>
  </div>
  <div class="card">
    <div class="card-label">Predicted Savings</div>
    <div class="card-value __SAVINGS_CLASS__">INR __PRED_SAVINGS__</div>
  </div>
  <div class="card">
    <div class="card-label">Model Performance</div>
    <div style="font-size:.9rem;margin-top:4px;">R2 <strong>__R2__</strong> | MAE <strong>__MAE__</strong></div>
    <div class="model-badge">__ALGO__</div>
  </div>
</div>

<div class="charts">
  <div class="chart-card">
    <h3>MONTHLY SPENDING VS BUDGET</h3>
    <canvas id="lineChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>SAVINGS TRAJECTORY</h3>
    <canvas id="barChart"></canvas>
  </div>
</div>

<div class="alerts">
  <h3 style="font-size:.85rem;color:var(--muted);margin-bottom:4px;">SYSTEM ALERTS</h3>
  __ALERTS__
</div>

<footer>
  Generated by Personal Finance Management System | __DATE__
</footer>

<script>
document.getElementById('ts').textContent = new Date().toLocaleString();

const labels = __LABELS__;
const spending = __SPENDING__;
const budget   = __BUDGET_ARR__;
const savings  = __SAVINGS_ARR__;

new Chart(document.getElementById('lineChart'), {
  type: 'line',
  data: {
    labels,
    datasets: [
      {
        label: 'Spending',
        data: spending,
        borderColor: '#ff4f6a',
        backgroundColor: 'rgba(255,79,106,.08)',
        tension: .3,
        fill: true,
        pointRadius: 4,
      },
      {
        label: 'Budget',
        data: budget,
        borderColor: '#6c63ff',
        borderDash: [6, 3],
        tension: 0,
        fill: false,
        pointRadius: 0,
      },
    ],
  },
  options: {
    plugins: { legend: { labels: { color: '#8892a4', font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: '#8892a4' }, grid: { color: '#2a2d3a' } },
      y: { ticks: { color: '#8892a4' }, grid: { color: '#2a2d3a' }, beginAtZero: true },
    },
  },
});

new Chart(document.getElementById('barChart'), {
  type: 'bar',
  data: {
    labels,
    datasets: [{
      label: 'Est. Savings',
      data: savings,
      backgroundColor: savings.map(v => v >= 0 ? 'rgba(62,207,142,.7)' : 'rgba(255,79,106,.7)'),
      borderRadius: 6,
    }],
  },
  options: {
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#8892a4' }, grid: { color: '#2a2d3a' } },
      y: { ticks: { color: '#8892a4' }, grid: { color: '#2a2d3a' } },
    },
  },
});
</script>
</body>
</html>
"""


def render_dashboard(report: dict, output_path: str = "dashboard.html") -> str:
    """Inject financial report data into the HTML template and write the file."""
    m = report["metrics"]
    cd = report["chart_data"]

    savings_class = "success" if m["predicted_savings_next_month"] >= 0 else "danger"

    alert_html = "\n".join(f'<div class="alert">{a}</div>' for a in report["alerts"])
    if not alert_html:
        alert_html = '<div class="alert ok">No anomalies detected. Financial health looks good.</div>'

    html = (
        DASHBOARD_HTML.replace("__CUR_SPEND__", str(m["current_monthly_spending"]))
        .replace("__BUDGET__", str(m["budget_limit"]))
        .replace("__PRED_SAVINGS__", str(m["predicted_savings_next_month"]))
        .replace("__SAVINGS_CLASS__", savings_class)
        .replace("__R2__", str(m["model"]["r2"]))
        .replace("__MAE__", str(m["model"]["mae"]))
        .replace("__ALGO__", m["model"]["algorithm"])
        .replace("__ALERTS__", alert_html)
        .replace("__DATE__", datetime.date.today().isoformat())
        .replace("__LABELS__", json.dumps(cd["labels"]))
        .replace("__SPENDING__", json.dumps(cd["spending"]))
        .replace("__BUDGET_ARR__", json.dumps(cd["budget"]))
        .replace("__SAVINGS_ARR__", json.dumps(cd["savings"]))
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return os.path.abspath(output_path)


# ============================================================
# SECTION 7 - JSON EXPORT
# ============================================================


def export_report(report: dict, output_path: str = "financial_report.json") -> str:
    """Serialize the financial report to JSON."""
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)
    return os.path.abspath(output_path)


# ============================================================
# SECTION 8 - MAIN ENTRY POINT & DATA SEEDER
# ============================================================


def create_demo_schema() -> None:
    """Create all tables needed for this standalone in-memory demo."""
    from django.db import connection

    logger.info("Creating Django tables in memory...")
    with connection.schema_editor() as editor:
        for model in (ContentType, Permission, Group, User, Expense, BudgetConfiguration):
            editor.create_model(model)


def seed_database() -> User:
    """
    Create an in-memory Django schema and populate it with five months of
    realistic expense and budget data for a demo user.
    """
    create_demo_schema()

    user = User.objects.create_user(username="demo_user", password="demo1234")

    logger.info("Seeding 5 months of expense & budget data...")
    categories = ["Food", "Transport", "Entertainment", "Utilities", "Shopping"]
    monthly_budgets = {
        "Food": Decimal("4000"),
        "Transport": Decimal("1500"),
        "Entertainment": Decimal("2000"),
        "Utilities": Decimal("1000"),
        "Shopping": Decimal("3000"),
    }
    base_amounts = {
        "Food": Decimal("800"),
        "Transport": Decimal("300"),
        "Entertainment": Decimal("450"),
        "Utilities": Decimal("250"),
        "Shopping": Decimal("600"),
    }

    for month_offset in range(5):
        month_date = datetime.date(2026, 1 + month_offset, 1)

        for cat in categories:
            BudgetConfiguration.objects.create(
                user=user,
                month=month_date,
                category=cat,
                budget_limit=monthly_budgets[cat],
            )
            for week in range(4):
                tx_date = month_date + datetime.timedelta(days=week * 7 + 2)
                variance = Decimal(
                    str(round(float(base_amounts[cat]) * 0.1 * (week % 3 - 1), 2))
                )
                Expense.objects.create(
                    user=user,
                    expense_date=tx_date,
                    amount=base_amounts[cat] + variance,
                    category=cat,
                    description=f"{cat} expense week {week + 1}",
                    payment_method="UPI",
                    currency="INR",
                )

    Expense.objects.create(
        user=user,
        expense_date=datetime.date(2026, 5, 22),
        amount=Decimal("2800.00"),
        category="Food",
        description="Catering - office party (anomalous)",
        payment_method="Card",
        currency="INR",
    )

    logger.info("Database seeding complete.")
    return user


def main() -> None:
    """Run the full pipeline: seed -> profile -> export -> dashboard."""
    logger.info("=" * 60)
    logger.info("Personal Finance Management System - Starting")
    logger.info("=" * 60)

    user = seed_database()

    logger.info("Generating financial profile for user '%s'...", user.username)
    report = generate_financial_profile(user)

    print("\n" + "=" * 60)
    print("  FINANCIAL REPORT (JSON)")
    print("=" * 60)
    print(json.dumps(report, indent=4, default=str))
    print("=" * 60 + "\n")

    json_path = export_report(report, "financial_report.json")
    logger.info("JSON report exported -> %s", json_path)

    html_path = render_dashboard(report, "dashboard.html")
    logger.info("Dashboard exported -> %s", html_path)
    logger.info("Open dashboard.html in any browser to view the UI.")

    logger.info("=" * 60)
    logger.info("System completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
