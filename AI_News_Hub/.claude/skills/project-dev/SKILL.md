---
name: project-dev
description: Working in the ai-news-hub project — a single-file vanilla HTML/CSS/JS app that aggregates AI news from multiple sources. Use whenever editing index.html, adding a news source, debugging a CORS issue, or shipping a UI change.
---

# Working in ai-news-hub

## When to use

Trigger this skill any time the user:
- Edits `index.html` (the only source file)
- Adds, removes, or fixes a news source
- Reports an empty / wrong category / wrong source on a card
- Asks about CORS, rate limits, or proxies
- Touches the filter UI (tabs, source chips, day window, search, download)
- Asks for a build / test command (there isn't one — see workflow)

## Repository overview

**One file. No build. No deps.** The entire app is `C:\Users\Eddie\workspace\ai-news-hub\index.html` (~1300 lines). It's a static SPA that runs by opening the file in a browser — no `package.json`, no `requirements.txt`, no Makefile, no CI, no Docker.

Stack: vanilla HTML + CSS custom properties (dark/light themed via `[data-theme]`) + plain ES2020+ JS. No framework, no bundler, no transpilation.

### Sections inside `index.html`

The JS is divided by banner comments `/* ── Section ── */`. Useful Grep target: `^/\* ──`. Key sections in order:

| Line ~ | Section            | Purpose                                                        |
|--------|--------------------|----------------------------------------------------------------|
| 403    | `Sources`          | `SOURCES` config — name, short label, brand color, badge bg    |
| 415    | `Category config`  | `CATS` — category keywords used by `categorize()`              |
| 453    | `Config`           | `REFRESH_MS`, `MIN_LAB_DAYS`, `PRESETS`, hour bounds           |
| 472    | `State`            | `windowHours`, `allStories`, `activeTab`, `activeSource`, etc. |
| 488    | `Helpers`          | `domain()`, `timeAgo()`, `sinceTs()`, `windowLabel()`, `esc()` |
| 554    | `Source fetchers`  | One async fn per source (HN, Anthropic, OpenAI, X, …)          |
| 873    | `Grid`             | `renderGrid()` — produces card HTML                            |
| 949    | `Meta row`         | `renderMeta()` — window selector lives here                    |
| 1028   | `Main fetch`       | `fetchNews()` — orchestrator (Promise.allSettled across sources) |
| 1105   | `Download / export`| `exportJSON()`, `exportCSV()`, `exportMarkdown()`              |
| 1258   | `Events`           | Refresh / theme / search / download menu / badge capture-phase |

## Common workflows

### Test a change
There is no test runner. The dev loop is:
1. Edit `index.html` with `Edit` (never rewrite the whole file with `Write` unless explicitly asked).
2. The Launch preview panel auto-reloads after a write — call out that "the change is visible in the preview panel" so the user knows to look.
3. For source-fetch debugging, instruct the user to open DevTools (F12) → Console. Every fetcher logs `[SourceName] N stories …`.

### Add a new news source

1. **Register it** in the `SOURCES` const (~line 403). Pick a brand color and a short label. Order matters for URL-dedup priority — put lab-specific sources first.
2. **Write `fetchXyz(since)`** in the source-fetchers section. It must return an array of records with these fields (mirror existing fetchers):
   ```js
   {
     id, title, url, discussionUrl,
     sourceKey,            // matches the SOURCES key
     points, comments,
     created_at_i,         // Unix seconds — required for sort + dedup
     author,
   }
   ```
3. **Wire it into the orchestrator** `fetchNews()` (~line 1028): add to the `Promise.allSettled` array, the destructured result variable, the `raw` spread, and the toast `nOk/N sources` counter.
4. **Update the loading message** (`Fetching from N sources…`).
5. **Console-log the count** so the user can verify in DevTools.

### Fix "source shows zero stories"

Symptoms → causes, in order to check:
- **No CORS headers** on the upstream API → route via `fetchThroughProxy()` (chain: corsproxy.io → allorigins.win → codetabs.com). arXiv is the existing example.
- **API restructured URLs** (e.g. OpenAI moved research from `/research/` to `/index/`) → broaden the Algolia `query` to just the bare domain, then tighten with a regex post-filter on `h.url`.
- **Window too narrow** for a low-cadence source → lab sources (Anthropic, OpenAI) enforce a 14-day floor via `MIN_LAB_DAYS`. Apply the same pattern: `Math.min(since, now - MIN_LAB_DAYS * 86400)`.
- **Title empty** → already filtered in the dedup step. Check DevTools console for the per-source count log.

## Coding conventions

- **Edit in place.** Use `Edit`, not `Write`. The single-file structure means a full rewrite drops anchor points the user has been pointing to in earlier turns.
- **Banner comments** (`/* ── X ── */`) are the architecture. New code goes inside an existing banner or under a new one.
- **HTML escaping is mandatory** for any user-derived string injected into `innerHTML`. Use the existing `esc()` helper. Story titles come from third-party APIs — they're untrusted.
- **No frameworks, no bundler, no JSX.** Plain `innerHTML` template strings. Event delegation on `document` for badge filters (capture phase — see `Badge filter clicks` section). Don't introduce React/Vue/etc.
- **CSS lives in `<style>`** at the top of the file with banner-comment sections. Theme colors are CSS custom properties on `:root[data-theme="…"]` — never hardcode `#fff` / `#000`.
- **Per-card link wrapping** uses an `<a>` around the entire card. Anything *inside* that should NOT navigate (badges, "HN ↗" link, format buttons) must `event.preventDefault()` + `event.stopPropagation()` in the **capture phase**, otherwise the `<a>` wins.
- **URL dedup ordering matters.** In `fetchNews()`, source-specific feeds (Anthropic, OpenAI, X) run *before* generic feeds (HN, Reddit) so their tagged version wins the URL-keyed dedup. Don't reorder casually.

## CORS / API gotchas

| API                           | CORS? | Notes                                                                  |
|-------------------------------|-------|------------------------------------------------------------------------|
| HN Algolia (`hn.algolia.com`) | ✅    | Free, no key. Use `search_by_date` for recency, not `search`.          |
| DEV.to (`dev.to/api`)         | ✅    | Use `tag=` filter + `state=fresh`.                                     |
| Reddit (`reddit.com/*.json`)  | ✅    | `raw_json=1` to get unescaped strings.                                 |
| Lobste.rs                     | ✅    | `/t/<tag>.json`.                                                       |
| arXiv (`export.arxiv.org`)    | ❌    | **Must** go through `fetchThroughProxy()`. Returns Atom XML.           |
| X / Twitter                   | ❌❌  | No public API since 2023. We surface tweets via HN Algolia URL search. |

## Validation checklist before finishing changes

- [ ] Used `Edit` (not `Write`) to modify `index.html`.
- [ ] Any string interpolated into `innerHTML` is wrapped in `esc()` if it could contain `<`, `>`, `&`, or `"`.
- [ ] If a new source was added: the count appears in the meta-row "N stories · M sources" *and* in the source-chip row *and* in the toast `X/N sources OK`.
- [ ] If a regex/path filter was changed: tested with a few sample URLs in DevTools console (`/<your-regex>/.test('https://...')`).
- [ ] If the user expects an immediate visual confirmation: explicitly told them "live in the preview panel" so they refresh / look.
- [ ] No introduction of build tooling, frameworks, npm/pip dependencies, or external script tags without explicit user request — the zero-dependency property is load-bearing.

## Windows-specific notes

- The repo lives at `C:\Users\Eddie\workspace\ai-news-hub`. Use backslash paths in `Bash` / `Read` / `Write` calls.
- `mkdir -p` works under the bash bundled with the harness — `New-Item -ItemType Directory -Force` is the PowerShell equivalent if needed.
- No build / run command exists; "running" the app is opening `index.html` in a browser (the Launch preview panel does this automatically on write).
