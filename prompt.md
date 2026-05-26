# Prompt: Build an AI-Powered Personal Finance Management System with ML Savings Prediction and Dashboard

# Domain
Financial Technology / Data Engineering / ML Engineering

# Task Overview
Imagine you're building a smart financial assistant that lives entirely inside a single Python script. It wakes up, creates its own database from scratch, fills it with five months of realistic spending data, learns from that data, makes predictions about future savings, flags anything suspicious, and then writes everything it found into a clean JSON report and a browser-ready dashboard — before quietly exiting. No server running in the background. No database files left on disk. No setup wizard. Just one script that does everything and leaves two useful files behind.
That is exactly what you are going to build.

⚙️ System Requirements

# Pipeline Architecture
Every time the script runs, it needs to follow this exact sequence — no skipping, no reordering. Start by bootstrapping Django with an in-memory database, then create all the tables on the fly, seed five months of demo data, pull everything into DataFrames, clean and engineer the features, train and select the best model, generate the financial profile, and finally write the JSON report and HTML dashboard to disk.
Four rules govern this flow. The machine learning step can never touch data that hasn't been cleaned first. Anomaly detection must always happen before budget alignment and lag features are calculated. The system must pick the best model using cross-validation whenever enough data exists, and fall back gracefully when it doesn't. The dashboard must only ever read from the final structured report dictionary — never directly from the database.

# Django ORM Configuration
Django must be configured entirely in code — no manage.py, no migration files, no external config. The database lives purely in memory, so nothing gets written to disk between runs. Every table is created at the start of each run and disappears the moment the script finishes. Think of it as assembling Django by hand — every switch flipped manually, every setting chosen deliberately. The secret key must carry a comment warning anyone reading the code that it needs to be replaced before this goes anywhere near production.

# Django Models
The system needs two models. The first captures individual spending transactions. Every record belongs to a user and stores the date the money was spent, the amount, which category it falls into, a short description, how the payment was made, and which currency was used. Categories cover the usual suspects — food, utilities, entertainment, transport, housing, healthcare, shopping, and a catch-all miscellaneous bucket. Payment methods cover UPI, card, cash, and net banking. Records should always come back in reverse chronological order so the most recent spending appears first.
The second model captures monthly budget targets. Each record ties a user to a specific category in a specific month and stores the spending limit for that combination. No two records can share the same user, month, and category — that combination must be unique across the entire table.

# ETL and Feature Engineering
This part of the system transforms raw database records into a clean, structured table that the machine learning engine can actually learn from. It should be organized as a class with exactly two public methods — one for cleaning the raw data and one for building the full feature set. Nothing else should be visible from the outside.
Cleaning is about making the data trustworthy before anything else happens. Any amount or budget figure that can't be parsed as a number should be replaced with zero rather than causing a crash. Dates need to be converted into proper date objects, and both DataFrames need a shared month column so they can be matched up when joined.
Building features is where the real transformation happens. If no expense records exist at all, the system should log a warning and stop immediately — there's nothing to work with. Otherwise, calculate the following for each user for each month, in this order:

Total spending — the sum of every transaction that month
Transaction count — how many individual purchases were made
Average daily spending — total spending spread evenly across thirty days
Anomaly count — how many transactions were statistically unusual that month, measured by how far each transaction strays from that user's personal average
Previous month spending — what the user spent the month before, so the model has a sense of trajectory
Budget limit — the planned ceiling for that month; if no budget was recorded, use 1.2× that month's actual spending as a reasonable stand-in
Target savings — the gap between the budget and what was actually spent; this is the number the model will learn to predict

A transaction counts as anomalous when its distance from the user's personal mean, measured in standard deviations, exceeds two. Any user whose transactions all happen to be identical will have a standard deviation of zero, so use a tiny floor value instead to avoid a division-by-zero error. Once the full feature table is ready, log its dimensions before returning it.

# ML Model Selection Engine
This part of the system decides which machine learning model is best suited to the available data and uses it to make the savings prediction. It should be organized as a class with exactly two public methods — one for training and selecting the model, and one for making predictions. All internal logic stays private.
Training and selection must adapt to how much data is actually available, because the right approach changes depending on dataset size:

When fewer than two months of data exist, there's genuinely nothing to learn from. The system should fall back to a predictor that always returns zero and log a clear warning explaining why.
When two to four months of data exist, training a single straightforward linear model is the right call. Cross-validation would be misleading with so few examples.
When five or more months of data exist, the system should run a proper competition between a simple linear model and a gradient boosting model. Whichever scores higher on cross-validated R-squared wins and becomes the champion. If the scores come back unusable for both, the linear model wins by default.

Keep the number of cross-validation folds small when data is limited — two folds when fewer than six rows exist, otherwise cap at three. The gradient boosting model must always use the same random seed so results are reproducible across runs. After selecting the champion, record its mean absolute error, mean squared error, R-squared score, and cross-validation R-squared score — all rounded to four decimal places.
Predicting takes a single row of features representing the most recent month and returns the estimated savings figure for next month. If someone tries to call this before training has happened, raise an error with a clear message explaining what went wrong.

 
#Financial Profile Generator
This function is the conductor of the whole system. It fetches the data, runs it through the pipeline, trains the model, makes the prediction, and assembles everything into the final report that gets written to disk.
If the expense records come back empty, raise an error immediately — there's no point continuing. Otherwise, run the full pipeline, generate the savings prediction for the most recent month, and calculate two additional figures: how far over budget the user is as a ratio, and a confidence score that reflects how certain the system is about the overspending flag.
Then check three conditions and add an alert for each one that's true:

If the user spent more than their budget, the alert should say by how much and include the confidence score
If the predicted savings for next month are negative, the alert should flag that the trajectory is heading in the wrong direction
If any anomalous transactions were detected in the most recent month, the alert should say how many were found

The final output is a single dictionary containing the user's ID and name, all the key metrics, the list of alerts, and the chart data needed to draw the dashboard visualizations.

#Error Handling and Defensive Programming

The system must never crash silently or produce output that looks correct but isn't. Every failure must be loud, specific, and caught at the right layer. Apply these rules without exception throughout the entire pipeline:
Never cast untrusted data directly to a number. Always route it through a safe conversion that replaces unparseable values with zero rather than raising an exception mid-pipeline. If the expense DataFrame comes back empty at any point after cleaning — not just before — raise a descriptive error immediately rather than letting the pipeline continue with nothing to work on. If predict() is called before train() has been run, raise a runtime error with a message that tells the caller exactly what they did wrong and what they need to do first. Every fallback path in the model selection logic must log a warning that names the fallback, explains why it was triggered, and states what the system will do instead — never fall back silently. Anomaly detection must handle the zero-standard-deviation edge case explicitly with a named floor constant rather than an inline magic number, and must log when it applies the floor so the behaviour is visible at runtime. If a budget record is missing for a given user-month combination, apply the 1.2× fallback and log which user and month triggered it — never apply the fallback silently as a default. If writing the JSON report or HTML dashboard to disk fails for any reason, catch the exception, log the full error, and re-raise it with a clear message so the caller knows the output files were not written. The overspending_flagged variable and the confidence score must both be fully computed and assigned before either is referenced anywhere else in the profile generator — reference-before-assignment is a runtime error that must be structurally impossible, not just avoided by convention.


# Constraints (All Must Be Met)

1. Single entry point.
Everything lives in one file. One command runs the whole system. No supporting Django files, no migration history, no separate configuration modules.

2. In-memory only.
The database exists only while the script is running. The moment it finishes, all data disappears. Every run is a completely fresh start.

3. Structured JSON output.
The report file must follow a precise structure — user identifiers at the top level, a metrics block with a nested model performance section, a flat list of alert strings, and a chart data block with four parallel arrays covering labels, spending, budget, and savings.

4. Deterministic results.
The gradient boosting model must use a fixed random seed. Given the same input data, the system must produce identical output on every run. The fallback order must never vary.

5. Self-contained dashboard.
The HTML file must work when opened directly in any browser. No local server, no build step, no locally hosted files. Everything it needs either comes from a CDN or is embedded directly in the file.

6. No unexplained numbers.
Every threshold used in the logic — whether it controls anomaly sensitivity, budget fallback sizing, or daily averaging — must be either named clearly or explained with a comment that says what assumption it encodes and why that value was chosen.

# File and Folder Structure
Keep it minimal and purposeful. Every file has one job and one job only:
project/
├── main.py                  ← the entire system lives here
├── requirements.txt         ← four dependencies, all pinned
├── .gitignore               ← keeps build artifacts and outputs out of version control
├── financial_report.json    ← written fresh on every run
└── dashboard.html           ← written fresh on every run

# Formatting Requirements

Every class and every public function must carry a proper docstring that explains what it does, what it expects as input, and what it returns. A one-line placeholder isn't enough.
Divide the file into eight clearly labelled sections using comment banners. Anyone reading the file should be able to find the part they care about without having to read everything first.
The ETL class may only expose two methods to the outside world. The ML engine class may only expose two methods to the outside world. Everything else stays private.
Never cast untrusted data directly to a number. Always route it through a safe conversion that replaces unparseable values with zero instead of crashing.
Every threshold and every magic-looking constant must live in one designated place near the top of its section — not buried inline wherever it first happened to be needed.

# Deliverables


A single fully working Python script that runs cleanly from a fresh virtual environment with no extra setup beyond installing the four dependencies.
A requirements file listing those four dependencies with minimum version pins.
A README section of no more than four hundred words that walks through how to set up the environment, how to run the script, what the eight sections of the pipeline do in plain English, how to open and read the dashboard, and how to push the project to GitHub.


# Evaluation Criteria

A good solution will do all of the following without being told twice:

Run the complete pipeline from database creation to dashboard output without a single error
Correctly identify the anomalous high-value transaction using Z-score logic
Only apply the budget fallback when a budget record is genuinely missing — never as a shortcut
Handle all three data-size scenarios in the model selection logic exactly as described
Produce a dashboard where every dynamic value is filled in correctly and savings figures are colored green or red based on their sign
Write docstrings that actually explain things rather than restate the function name
Handle bad data, missing values, and empty inputs gracefully without crashing
