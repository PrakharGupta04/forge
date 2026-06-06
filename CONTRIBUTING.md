# Contributing to Forge

Thank you for contributing to Forge.

## Adding a New Metric

1. Create a new file in `forge/metrics/` (for example `your_metric_name.py`).
2. Subclass `BaseMetric` from `forge.metrics.base`.
3. Implement the required metric logic.
4. Implement `score_with_explanation()` and return a `MetricResult`.
5. Register the metric in `ALL_METRICS` inside `forge/metrics/engine.py`.
6. Add unit tests in `tests/unit/`.
7. Update the README if the metric introduces new functionality.

## Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v
```

## Development Setup

Follow the Quick Start section in `README.md`.

## Pull Requests

Before opening a pull request:

* Run all unit tests.
* Ensure new metrics include tests.
* Update documentation where required.
* Keep metric behavior explainable and deterministic where possible.
* Do not make undocumented breaking changes.
