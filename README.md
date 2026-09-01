# Employee Scheduling Test (OR-Tools CP-SAT Prototype)

A minimal, standalone Python script that proves out **Google OR-Tools CP-SAT**
for shift scheduling. This is Step 1 of a larger AI-driven factory
production scheduling project — no web app, database, API, frontend, or LLM
involved yet. Just the core constraint-solving logic.

Adapted from Google's official example:
https://developers.google.com/optimization/scheduling/employee_scheduling

## Project structure

```
employee-scheduling-test/
├── data/
│   └── schedule_data.json   # workers, days, shifts, availability
├── scheduler.py              # loads data, builds/solves the CP-SAT model
├── requirements.txt
└── README.md
```

## Setup

```bash
cd employee-scheduling-test
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python scheduler.py
```

You'll see a printed schedule table, followed by basic statistics
(total shifts, shifts per worker, any unassigned shifts), and the
solver's status (OPTIMAL / FEASIBLE / INFEASIBLE).

## Scenario modeled

- 4 workers (A, B, C, D)
- 5 days (Monday–Friday)
- 2 shifts/day (Morning 08:00–16:00, Evening 16:00–00:00) → 10 shifts total
- Rules:
  1. Each shift needs exactly one worker.
  2. No worker can work two shifts in the same day.
  3. No worker works more than 5 shifts in the week.
  4. Shifts are distributed as evenly as possible across workers.
  5. Worker A is unavailable all day Wednesday.
  6. Worker B is unavailable Monday morning.
  7. Worker C is unavailable Friday evening.
  8. Worker D is unavailable Tuesday evening.

## Next step

Once this is validated, the same modeling pattern (boolean decision
variables + CP-SAT constraints) will be extended to factory scheduling:
machines, work orders, and operators instead of just workers and shifts.
