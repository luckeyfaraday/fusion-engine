# Contributing

## Setup

```bash
python3 -m pip install -e ".[server,test]"
```

## Checks

Run the offline checks before opening a pull request:

```bash
python3 -m py_compile cli.py fusion.py panels.py server.py evals/benchmarks.py evals/graders.py evals/prepare.py evals/report.py evals/run_eval.py evals/systems.py
python3 cli.py panels
pytest -q
```

Do not commit `.env`, generated benchmark datasets, eval results, caches, or
local Context Workspace state. They are ignored by default.
