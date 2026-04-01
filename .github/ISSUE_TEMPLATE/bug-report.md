---
name: Bug Report
about: Report a bug, crash, or unexpected behavior in Quill-Sort’s source code.
title: "[BUG]"
labels: ''
assignees: dragonbreathIT

---

**Describe the bug**
A clear and concise description of what the bug is.
Example: “`quill_sort()` produces a different order than `sorted()` for this dataset…”

---

**To Reproduce**
Steps to reproduce the behavior (ideally with a small, copy-pastable code sample):

1. Example: create a small dataset
2. Call `quill_sort(...)` or `quill_sorted(...)`
3. Call Python’s built-in `sorted(...)` with the same arguments
4. Show how the results differ or where an error occurs

If possible, include a minimal code snippet:

```python
from quill import quill_sort, quill_sorted

data = [/* small failing sample */]
expected = sorted(data, key=..., reverse=...)
quill_sort(data, key=..., reverse=...)

print("quill :", data)
print("python:", expected)
```

---

**Expected behavior**
A clear and concise description of what you expected to happen.
Example: “Result should be identical to `sorted(data, key=..., reverse=...)`.”

---

**Environment (please complete the following information):**

- Quill-Sort version: e.g. 4.0.4
- Python version: e.g. 3.13.12
- OS and CPU: e.g. Windows 11, 28-core CPU, 16 GB RAM
- Mode used: `high_performance_mode=True/False`, `parallel=True/False`
- Approximate data size: e.g. 100K ints, 10M floats, 1B ints

---

**Screenshots / logs (optional)**
If helpful, add any error tracebacks, console logs, or screenshots here.

---

**Additional context**
Add any other context about the problem here.
Example: “This only happens when using external mode with `.qwrite` files,” or “Only fails on very high-duplicate integer datasets.”
