"""Manual verification for the Forge benchmark dataset.

Confirms the structural integrity and strengthened constraints of the
50-task benchmark suite distributed across five domains:

* All 50 tasks load and pass ``BenchmarkLoader._validate_task``.
* Every domain contains exactly 10 tasks.
* Every ``task_id`` is unique across the full dataset.
* Every ``golden_trajectory`` contains at least one well-formed step.
* Every ``multi_turn`` task includes a ``conversation_history`` field.

Lives in ``scripts/`` so pytest's discovery (``testpaths = tests``) does
not pick it up. Run with::

    python scripts/verify_benchmark_dataset.py
"""

from __future__ import annotations

from collections import Counter

from forge.benchmark.loader import BenchmarkLoader


_EXPECTED_DOMAINS: tuple[str, ...] = (
    "factual_research",
    "data_analysis",
    "code_tasks",
    "tool_recovery",
    "multi_turn",
)


def main() -> None:
    loader = BenchmarkLoader()
    print(f"Data directory: {loader.data_dir}")

    discovered = loader.domains()
    print(f"\nDiscovered domains ({len(discovered)}): {discovered}")
    missing_domains = set(_EXPECTED_DOMAINS) - set(discovered)
    extra_domains = set(discovered) - set(_EXPECTED_DOMAINS)
    assert not missing_domains, f"missing domain subdirectories: {sorted(missing_domains)}"
    assert not extra_domains, f"unexpected domain subdirectories: {sorted(extra_domains)}"

    all_tasks = loader.load()
    total = len(all_tasks)

    per_domain = Counter(t["domain"] for t in all_tasks)
    print("\nTasks per domain:")
    for dom in _EXPECTED_DOMAINS:
        c = per_domain.get(dom, 0)
        status = "OK" if c == 10 else "FAIL"
        print(f"  {dom:<20} {c:>3}  [{status}]")
        assert c == 10, f"domain {dom!r} has {c} tasks, expected exactly 10"

    print(f"\nTotal tasks: {total}")
    assert total == 50, f"expected exactly 50 total tasks, got {total}"

    print("\n--- First task per domain (structure preview) ---")
    for dom in _EXPECTED_DOMAINS:
        first = next(t for t in all_tasks if t["domain"] == dom)
        task_preview = first["task"].replace("\n", " ")
        if len(task_preview) > 100:
            task_preview = task_preview[:97] + "..."
        print(f"\n[{dom}] {first['task_id']} ({first['difficulty']})")
        print(f"  task          : {task_preview}")
        print(f"  ground_truth  : {first['ground_truth']}")
        print(f"  required_tools: {first['required_tools']}")
        print(f"  steps         : {len(first['golden_trajectory']['steps'])} step(s)")
        if dom == "multi_turn":
            print(f"  conv turns    : {len(first['conversation_history'])}")

    print("\n--- Strengthened constraint checks ---")

    ids = [t["task_id"] for t in all_tasks]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    assert not duplicates, f"duplicate task_ids: {duplicates}"
    print(f"  task_id uniqueness                : OK  ({len(ids)} unique ids)")

    for t in all_tasks:
        n_steps = len(t["golden_trajectory"]["steps"])
        assert n_steps >= 1, f"{t['task_id']}: golden_trajectory has 0 steps"
    print(f"  every golden_trajectory non-empty : OK")

    for t in all_tasks:
        loader._validate_task(t, f"<in-memory:{t['task_id']}>")
    print(f"  every task passes _validate_task  : OK")

    for t in all_tasks:
        if t["domain"] == "multi_turn":
            assert "conversation_history" in t, (
                f"{t['task_id']}: multi_turn task is missing conversation_history"
            )
            conv = t["conversation_history"]
            assert isinstance(conv, list) and len(conv) >= 3, (
                f"{t['task_id']}: conversation_history should have at least 3 turns, "
                f"got {len(conv) if isinstance(conv, list) else type(conv).__name__}"
            )
    print(f"  multi_turn tasks have conv_history: OK")

    difficulty_counter = Counter(t["difficulty"] for t in all_tasks)
    print(f"\nDifficulty distribution: {dict(difficulty_counter)}")
    for d in difficulty_counter:
        assert d in {"easy", "medium", "hard"}, f"unexpected difficulty: {d!r}"

    print("\nBenchmark dataset verified: 50 tasks across 5 domains")


if __name__ == "__main__":
    main()
