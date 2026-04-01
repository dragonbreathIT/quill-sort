---
name: Immediate Issue
about: Use this template to report a bug requiring immediate attention.
title: "[EMERGENCY]"
labels: Bug (Major)
assignees: dragonbreathIT

---

⚠️ **Critical / Emergency Issue**

> Use this template only for _severe_ problems (data corruption, totally wrong
> results, crashes that make Quill-Sort unusable, etc.).

---

**What is the critical issue?**  
Describe exactly what went wrong and why this feels like an emergency.  
Example: “Quill-Sort returns a different order than `sorted()` for simple int data,” or “External mode corrupted output / lost data.”

---

**Minimal code + dataset that reproduces it**  
Share the smallest example that still reproduces the issue.

```python
from quill import quill_sort, quill_sorted

data = [/* small failing sample */]
expected = sorted(data, key=..., reverse=...)

quill_sort(data, key=..., reverse=...)

print("quill :", data)
print("python:", expected)
```

If it only happens on large datasets, describe the shape/size (e.g. “100M int32, external mode”).

---

**How severe is the impact?**  
Check all that apply:

- [ ] Results are **wrong** compared to `sorted()`
- [ ] Application **crashes** or hangs
- [ ] **Data may be lost or corrupted**
- [ ] External mode fills disk or leaves many `.qwrite` files
- [ ] Quill-Sort becomes unusable for my workload

Add any details about how this affects you (e.g. “production pipeline blocked”, “benchmarks invalid”, etc.).

---

**Environment details**

- Quill-Sort version: `e.g. 4.0.4`
- Python version: `e.g. 3.13.12`
- OS and CPU: `e.g. Windows 11, 28-core CPU, 16 GB RAM`
- Modes used: `high_performance_mode=True/False`, `parallel=True/False`
- Approximate data size and type: `e.g. 1B int32, 10M floats, 5M strings`

---

**Logs, traces, and external-mode output**  
Paste any stack traces, console output, or external-mode prompts (e.g. `.qwrite` info, “Authorize external sort?” messages).

---

**Anything else that might help debug this fast?**  
Mention if:

- It started after a specific version upgrade.
- It goes away if you disable `parallel` or `high_performance_mode`.
- It only happens on a certain OS / machine.
