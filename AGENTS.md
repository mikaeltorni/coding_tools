# AGENTS.md — coding_tools

Project instructions for any agent working in this repository. They outrank
generic agent defaults and any skill, including discoverability/SEO skills.

## What this repository is

A research repository for commit-message generation and evaluation: dataset tooling (`data/`), fine-tuning experiments (`finetuning/`), evaluation harnesses (`evals/`), and the `src/` package behind `main.py`.

## Never add a static docs site

This repository is read on GitHub itself; it is not published as a website.
Do not create — and do not restore — any of:

- `docs/index.html` or any other HTML page,
- `sitemap.xml` or `robots.txt` anywhere in the tree,
- `_config.yml`, `.nojekyll`, Jekyll/Pages scaffolding, or a GitHub Pages
  deployment,
- README badges or links that point at a `github.io` site.

A missing published site is not a gap to fix. SEO or discoverability audits
score the published-site and crawlability rows
`N/A — static docs site out of scope by owner policy` and compute the
normalized total without them. Never enable GitHub Pages here.

## Never add an `llms.txt`

No `llms.txt` belongs in this repository — not at the root and not under
`docs/`. Never add or restore one. The README and the GitHub About text carry
the one-line definitional sentence on their own; audits score the `llms.txt`
row `N/A — llms.txt out of scope by owner policy` and evidence the
definitional-sentence row from the README and About text only.

## Repository assets are not a site

`docs/commit-flow.svg` is the flow diagram. These are repository assets that the README
embeds — keep them.

## No CI

Do not add `.github/workflows/` or any other CI/CD pipeline, and do not add a
build-status badge. Verification runs locally.

## Never add community-process files

This repository follows the owner-wide SEO policy for public repositories. Do
not add or restore `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
`.github/ISSUE_TEMPLATE/`, or `.github/PULL_REQUEST_TEMPLATE.md` (or equivalent
pull-request templates). These community and contribution workflows are out of
scope for this research and evaluation repository; keep project-specific
licensing and citation metadata instead.
