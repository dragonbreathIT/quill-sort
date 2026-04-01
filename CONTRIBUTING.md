# Contributing to Quill-Sort

Thanks for your interest in Quill-Sort! The project is still under heavy
development, so things may change quickly.

## Issues

- Use GitHub Issues to report bugs and request features.
- When reporting a bug, please include:
  - Python version (e.g. 3.13.12)
  - OS and CPU details
  - Quill-Sort version (`python -m quill --version` if available)
  - A minimal reproducible example
  - Use a tag to help us find your issue faster (view "Issue labels" below for more information)

## Issue labels

- **Bug (Major)**  
  Something broke in a serious way and needs to be fixed as soon as possible.

- **Bug (Minor)**  
  Something isn’t working correctly, but it’s not critical.

- **Documentation**  
  Improvements, clarifications, or additions to docs and examples.

- **Duplicate**  
  This issue already exists elsewhere and will usually be closed in favor of the original.

- **Enhancement**  
  New feature, improvement, or change request. Might be accepted, might be parked.

- **First Issue**  
  Great for first-time contributors. Welcome to the Quill-Sort community!

- **Help Wanted**  
  I’m explicitly looking for extra help or ideas on this.

- **Incorrect**  
  Behavior, benchmark, or documentation that appears to be wrong or misleading.

- **Question**  
  “How do I…?” or “Why does Quill-Sort do X?” type issues.

- **Shitpost**  
  Self-explanatory. For memes, jokes, or non-serious threads.

- **Won’t Fix**  
  Acknowledged, but this won’t be worked on (by design, too risky, or out of scope).

## Pull Requests

- For now, please open an issue _before_ large feature PRs so we can discuss
  the design.
- Try to include tests for any bugfix or new behavior.
- Make sure `python -m pytest` (or your test command) passes locally.

I’m still stabilizing the API and internals, so I may be opinionated about
what gets merged. Don’t take review comments personally 🙂
