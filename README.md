
  PERSONAL FINANCE MANAGEMENT SYSTEM
  AI-Powered Savings Prediction and Dashboard


ABOUT
-----
A standalone Python script that boots a Django database from
scratch, seeds five months of realistic spending data, trains
a machine learning model, detects unusual transactions, and
writes a JSON report and browser-ready HTML dashboard to disk.
One command. No setup wizard. No files left behind except the
two outputs.


FEATURES
--------
  - In-memory Django ORM (nothing written to disk)
  - Demo expense and budget seeding (5 months)
  - Monthly spending feature engineering
  - Z-score anomaly detection
  - ML savings prediction with automatic model selection
  - Structured JSON report export
  - Browser-ready dashboard.html with Chart.js graphs


HOW TO RUN (VS Code)
--------------------
Step 1  Create a virtual environment

        python -m venv .venv

Step 2  Activate it

        .\.venv\Scripts\Activate.ps1

Step 3  Install dependencies

        pip install -r requirements.txt

Step 4  Run the script

        python main.py

After running, two files are created in the project folder:

        financial_report.json
        dashboard.html

Open dashboard.html in any browser to view the dashboard.
No local server needed.


WHAT THE 8 PIPELINE SECTIONS DO
--------------------------------
  1. Configuration     Boots Django entirely in code using an
                       in-memory SQLite database. Nothing is
                       written to disk between runs.

  2. Models            Defines two database tables: one for
                       individual spending transactions and one
                       for monthly budget targets.

  3. Seeding           Fills the database with five months of
                       realistic demo data plus one high-value
                       anomaly to test detection logic.

  4. ETL               Pulls records into DataFrames, coerces
                       types safely, and builds the feature
                       table the ML engine learns from.

  5. ML Engine         Picks the best model for the available
                       data size, runs cross-validation when
                       enough data exists, and falls back
                       gracefully when it does not.

  6. Profile           Orchestrates the full pipeline, generates
                       the savings prediction, computes the
                       overspending confidence score, and
                       assembles the final report dictionary.

  7. JSON Report       Writes the structured report to
                       financial_report.json on every run.

  8. Dashboard         Renders the HTML file with Chart.js
                       graphs. Every dynamic value is injected
                       from the report dictionary. Savings
                       figures are coloured green or red based
                       on their sign.


UPLOAD TO GITHUB
----------------
  git init
  git add .
  git commit -m "Initial personal finance dashboard project"
  git branch -M main
  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
  git push -u origin main

Replace YOUR_USERNAME and YOUR_REPO with your GitHub details.


================================================================
