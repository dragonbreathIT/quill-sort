# Releasing the companion accelerators

The five companion packages are published as **binary wheels for every platform**
(Linux x86_64/aarch64, macOS x86_64/arm64, Windows) plus an sdist, built in CI by
[`.github/workflows/companion-wheels.yml`](../.github/workflows/companion-wheels.yml)
with [cibuildwheel](https://cibuildwheel.pypa.io/).

## One-time setup: PyPI Trusted Publishing (no token in CI)

For each of the five projects — `quill-fastsort`, `quill-fastsort-parallel`,
`quill-fastsort-ips4o`, `quill-fastsort-numa`, `quill-fastsort-simd` — on PyPI:

1. Open the project → **Manage → Publishing → Add a new publisher → GitHub**.
2. Fill in:
   - **Owner:** your GitHub user/org
   - **Repository:** this repo's name
   - **Workflow name:** `companion-wheels.yml`
   - **Environment:** *(leave blank)*
3. Save.

That authorizes this workflow to publish that project via short-lived OIDC
tokens — nothing secret is stored in the repo or in GitHub secrets.

> **Token alternative.** If you'd rather use an API token: add it as the repo
> secret `PYPI_API_TOKEN`, then in the workflow's `publish` job uncomment the
> `password:` line and remove the `id-token: write` permission.

## Cutting a release

1. Bump the `version` in each companion you're releasing
   (`companions/<pkg>/pyproject.toml` **and** its `__init__.py` `__version__`).
   PyPI refuses to overwrite an existing version, so every publish needs a new
   number.
2. Commit and push.
3. On GitHub, **Releases → Draft a new release**, create a tag (e.g.
   `companions-v2025.07`), and **Publish release**.
4. The workflow builds + smoke-tests wheels on all platforms, builds each sdist,
   and publishes each package to PyPI via Trusted Publishing. `skip-existing:
   true` means re-runs won't fail on versions already uploaded.

To dry-run without publishing, open a PR touching `companions/**` (builds +
tests, no publish) or use **Actions → Companion wheels → Run workflow**
(`workflow_dispatch` does publish — use a PR for a pure dry run).

## What gets built

Per package, per Python 3.8–3.13:

| Platform | Runner | Wheel tags |
|----------|--------|-----------|
| Linux x86_64  | `ubuntu-latest`    | `manylinux_x86_64`, `musllinux_x86_64` |
| Linux aarch64 | `ubuntu-24.04-arm` | `manylinux_aarch64`, `musllinux_aarch64` |
| macOS x86_64  | `macos-13`         | `macosx_x86_64` |
| macOS arm64   | `macos-14`         | `macosx_arm64` |
| Windows       | `windows-latest`   | `win_amd64` |

Native runners build each architecture directly (no QEMU emulation). The sort
kernels are header-only C++17, so no per-wheel system libraries are required.
Only `quill-fastsort-numa` has platform-specific code (`detect_topology()` reads
Linux `sysfs` under `#ifdef __linux__`; elsewhere it returns `None`).
