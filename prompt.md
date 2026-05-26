Build an AI-Powered Personal Finance Management System
Domain: Financial Technology · Data Engineering · ML Engineering

Overview
A single Python script that boots its own database, seeds five months of spending data, trains a machine learning model, detects unusual transactions, and writes a JSON report and HTML dashboard to disk — then exits cleanly. No server. No leftover files. One command does everything.

Input Data
The system seeds all data at runtime. No external files required.
Expense Transactions (one per week per category, five months)
user_id — demo Django user created at runtime
expense_date — one transaction per week per category
amount — base amount with small weekly variance
category — Food · Transport · Entertainment · Utilities · Shopping
description — plain-text label per transaction
payment_method — UPI · Card · Cash · NetBanking
currency — INR

Monthly Budget Records (one limit per category per month)

month — January to May 2026
budget_limit — Food 4000 / Transport 1500 / Entertainment 2000 / Utilities 1000 / Shopping 3000

Anomalous Transaction (injected into May 2026)

INR 2800 Food transaction labelled as office party catering
Triggers Z-score detection and appears in the alert output


Output
Two files written fresh on every run.
financial_report.json
user_id and username at the top level
metrics block — current spending, budget limit, predicted savings, overspending flag, confidence score
model block — algorithm name, MAE, MSE, R², CV R² (all 4 decimal places)
alerts — flat list of plain-English triggered conditions
chart_data — four parallel arrays: labels, spending, budget, savings

dashboard.html

Opens in any browser with no server required
Line chart — monthly spending vs budget limit
Bar chart — savings trajectory (green = positive, red = negative)
Stat cards — current spending, budget, predicted savings, model name + scores
Alert panel — one coloured box per triggered condition
All values injected from the report dictionary only — never from the database


System Requirements

Pipeline Sequence (must run in this exact order)
Bootstrap Django with in-memory database
Create tables and seed five months of demo data
Pull records into DataFrames and clean
Engineer features
Train and select the best model
Generate financial profile
Write JSON report and HTML dashboard to disk

Django Setup

Configured entirely in code — no manage.py, no migration files
In-memory SQLite only — nothing written to disk between runs
Secret key must carry a warning comment about production use

Django Models

Expense — stores user, date, amount, category, description, payment method, currency. Orders by most recent first.
BudgetConfiguration — stores user, month, category, budget limit. Unique constraint on user + month + category.

ETL and Feature Engineering
Two public methods only — clean() and build_features(). Everything else private.
Cleaning:

Parse amounts and budget figures safely — unparseable values become zero
Convert dates to date objects
Add a shared month column to both DataFrames

Features built per user per month (in this order):

total_spending — sum of all transactions
transaction_count — number of purchases
avg_daily_spending — total divided by 30
anomaly_count — transactions more than 2 standard deviations from the user's mean
prev_month_spending — lag feature from the previous month
budget_limit — from budget records; fallback is 1.2× actual spending if missing
target_savings — budget minus spending (the prediction target)

ML Model Selection Engine
Two public methods only — train() and predict(). Everything else private.
Selection logic by data size:

Fewer than 2 months — return zero, log a warning
2 to 4 months — train LinearRegression only, no cross-validation
5 or more months — compete LinearRegression vs GradientBoosting; highest CV R² wins

Additional rules:

Cross-validation folds:
 2 if fewer than 6 rows, otherwise 3
GradientBoosting must use a fixed random seed
Record MAE, MSE, R², CV R² — all rounded to 4 decimal places
Calling predict() before train() raises a descriptive error

Financial Profile Generator
Orchestrates the full pipeline and returns one report dictionary containing user identifiers, all metrics, the alerts list, and chart data.

Three alert conditions:

Spending exceeds budget — include overspend percentage and confidence score
Predicted savings are negative — flag the trajectory
Anomalous transactions detected — state how many


Error Handling

Unparseable amounts → replaced with zero, never crash
Empty DataFrame after cleaning → raise a descriptive error immediately
predict() before train() → raise a runtime error with a clear message
Every fallback → log the reason, what triggered it, and what happens next
Zero standard deviation in anomaly detection → use a named floor constant, log when applied
Missing budget record → apply 1.2× fallback and log the user and month
File write failure → catch, log, and re-raise with a clear message
overspending_flagged and confidence score → must be fully assigned before any reference


Constraints

Single file — everything lives in main.py, one command runs the system
In-memory only — database disappears when the script exits
Deterministic — fixed random seed, identical output for identical input
Self-contained dashboard — works in any browser, no local server, CDN only
No magic numbers — every threshold named or commented with its reasoning


File Structure
project/
 .main.py                ← entire system
 .requirements.txt       ← four pinned dependencies
 .gitignore
 .financial_report.json  ← written on every run
 .dashboard.html         ← written on every run

Formatting Requirements

Every class and public function carries a full docstring — not a one-liner
File divided into eight clearly labelled sections with comment banners
ETL class: two public methods maximum
ML engine class: two public methods maximum
All constants and thresholds defined at the top of their section — never inline


Deliverables

main.py — runs cleanly from a fresh virtual environment
requirements.txt — four dependencies with minimum version pins
README under 400 words covering: setup, run instructions, eight pipeline sections, dashboard usage, GitHub upload steps


Evaluation Criteria

Runs from database creation to dashboard output without a single error
Correctly flags the anomalous transaction using Z-score logic
Budget fallback applied only when a record is genuinely missing
All three data-size scenarios handled exactly as specified
Dashboard values correct; savings coloured green or red by sign
Docstrings explain behaviour — not just restate the function name
Bad data, missing values, and empty inputs handled without crashing
No variable referenced before it is fully assigned
No silent fallbacks — every fallback logs its reason
File write failures surface a clear error to the caller
