### Build an AI-Powered Personal Finance Management System Domain: Financial Technology · Data Engineering · ML Engineering

Overview A single Python script that boots its own database, seeds five months of spending data, trains a machine learning model, detects unusual transactions, and writes a **JSON** report and **HTML** dashboard to disk — then exits cleanly. No server. No leftover files. One command does everything.

### Input Data

The system seeds all data at runtime. No external files required. Expense Transactions (one per week per category, five months)

* user_id — demo Django user created at runtime 
* expense_date — one transaction per week per category
* amount — base amount with small weekly variance 
* category — Food · Transport · Entertainment · Utilities ·
* Shopping description — plain-text label per transaction payment_method — **UPI** · Card · Cash · NetBanking currency — **INR**

Monthly Budget Records (one limit per category per month)

month — January to May **2026** budget_limit — Food **4000** / Transport **1500** / Entertainment **2000** / Utilities **1000** / Shopping **3000**

Anomalous Transaction (injected into May **2026**)

**INR** **2800** Food transaction labelled as office party catering Triggers Z-score detection and appears in the alert 
### Output
Output Two files written fresh on every run. financial_report.json

* user_id and username at the top level 
* metrics block — current spending, budget limit, predicted savings, overspending flag, confidence score model block — algorithm name, **MAE**, **MSE**, R², CV R² (all 4 decimal places)
* alerts — flat list of plain-English triggered conditions chart_data — four parallel arrays: labels, spending, budget, savings

dashboard.html

Opens in any browser with no server required 
* Line chart — monthly spending vs budget limit 
* Bar chart — savings trajectory (green = positive, red = negative)
* Stat cards — current spending, budget, predicted savings, model name + scores Alert panel — one coloured box per triggered condition All values injected from the report dictionary only — never from the database

### System Requirements

Pipeline Sequence (must run in this exact order)

Bootstrap Django with in-memory database Create tables and seed five months of demo data Pull records into DataFrames and clean Engineer features Train and select the best model Generate financial profile Write **JSON** report and **HTML** dashboard to disk

### Django Setup

Configured entirely in code — no manage.py, no migration files In-memory SQLite only — nothing written to disk between runs Secret key must carry a warning comment about production use

### Django Models

* Expense — stores user, date, amount, category, description, payment method, currency. 
* Orders by most recent first. *
* BudgetConfiguration — stores user, month, category, budget limit. *
* Unique constraint on user + month + category.

**ETL** and Feature Engineering Two public methods only — clean() and build_features(). Everything else private. Cleaning:

Parse amounts and budget figures safely — unparseable values become zero Convert dates to date objects Add a shared month column to both DataFrames

### Features built per user per month (in this order):

* total_spending — sum of all transactions 
* transaction_count — number of purchases 
* avg_daily_spending — total divided by 30 
* anomaly_count — transactions more than 2 standard deviations from the user's mean 
* prev_month_spending — lag feature from the previous month 
* budget_limit — from budget records; fallback is 1.2× actual spending if missing 
* target_savings — budget minus spending (the prediction target)

ML Model Selection Engine Two public methods only — train() and predict(). Everything else private. Selection logic by data size:

Fewer than 2 months — return zero, log a warning 2 to 4 months — train LinearRegression only, no cross-validation 5 or more months — compete LinearRegression vs GradientBoosting; highest CV R² wins

Additional rules:

Cross-validation folds: 2 if fewer than 6 rows, otherwise 3 GradientBoosting must use a fixed random seed Record **MAE**, **MSE**, R², CV R² — all rounded to 4 decimal places Calling predict() before train() raises a descriptive error

### Financial Profile Generator

Orchestrates the full pipeline and returns one report dictionary containing user identifiers, all metrics, the alerts list, and chart data.

Three alert conditions:

Spending exceeds budget — include overspend percentage and confidence score Predicted savings are negative — flag the trajectory Anomalous transactions detected — state how many

### Error Handling

* Unparseable amounts → replaced with zero, never crash 
* Empty DataFrame after cleaning → raise a descriptive error immediately 
* predict() before train() → raise a runtime error with a clear message 
* Every fallback → log the reason, what triggered it, and what happens next 
* Zero standard deviation in anomaly detection → use a named floor constant, log when applied
* Missing budget record → apply 1.2× fallback and log the user and month
* File write failure → catch, log, and re-raise with a clear message overspending_flagged and confidence score → must be fully assigned before any reference

 ### Constraints

* Single file — everything lives in main.py, one command runs the system
* In-memory only — database disappears when the script exits
* Deterministic — fixed random seed, identical output for identical input 
* Self-contained dashboard — works in any browser, no local server, **CDN** only 
* No magic numbers — every threshold named or commented with its reasoning

### File Structure
1. main.py
Contains the entire Personal Finance Management System, including Django implementation, ETL Process, Machine Learning algorithms, anomaly detection, reporting, and dashboards.
2. requirements.txt
Lists all required packages for the application, including Django, pandas, scikit-learn, and matplotlib.
3. .gitignore
Excludes irrelevant files, such as virtual environments, cache files, and outputs, from being pushed on GitHub.
4. financial_report.json
Generates a new file automatically each time the application runs; stores data regarding financial metrics, alerts, savings estimates, and machine learning models evaluation.
5. dashboard.html
Automatically generated interactive dashboard showing spending patterns, savings estimates, alerts, and financial analytics through the use of Chart.js.





### Formatting Requirements

- Complete Documentation Standards → Every class and public function must contain a detailed docstring explaining purpose, behavior, parameters, return values, assumptions, and limitations instead of a single-line description.
- Structured File Organization → The `main.py` file must be divided into eight clearly labelled sections using descriptive comment banners for better readability and maintainability.
- Restricted Public Interfaces → The ETL class should expose a maximum of two public methods, and the ML engine class should also expose only two public methods to maintain clean abstraction boundaries.
- Centralized Constants Management → All thresholds, fallback values, configuration constants, and tuning parameters must be declared at the top of their respective sections rather than being hardcoded inline.
- Required Deliverables → The project must include:
  - `main.py` → fully runnable from a fresh virtual environment
  - `requirements.txt` → contains four dependencies with minimum version pins
  - `README.md` → under 400 words covering setup instructions, execution steps, eight pipeline stages, dashboard usage, and GitHub upload guidance

### Evaluation Criteria

- End-to-End Execution → The system must run successfully from database initialization to dashboard generation without producing any runtime errors.
- Accurate Anomaly Detection → The injected anomalous transaction must be correctly identified using Z-score based detection logic.
- Smart Budget Fallback → Budget fallback values should only be applied when a genuine budget record is missing.
- Dynamic Model Handling → All three data-size scenarios must be handled exactly according to the specified ML selection rules.
- Correct Dashboard Visualization → Dashboard charts and metrics must display accurate values, with savings highlighted in green for positive values and red for negative values.
- Meaningful Documentation → Docstrings should clearly explain the purpose, behavior, assumptions, and limitations of functions rather than simply repeating function names.
- Robust Data Handling → Invalid data, missing records, and empty datasets must be handled gracefully without crashing the application.
- Safe Variable Initialization → No variable should ever be referenced before being completely assigned and initialized.
- Transparent Fallback Logging → Every fallback mechanism must explicitly log why it occurred and what corrective action was taken.
- Clear File Error Reporting → Any failure during file generation or writing must surface a descriptive error message to the caller.
