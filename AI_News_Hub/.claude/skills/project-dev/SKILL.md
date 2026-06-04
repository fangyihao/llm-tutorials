---
name: project-dev
description: Working in the ai-news-hub project — a single-file vanilla HTML/CSS/JS app that aggregates AI news from 9 sources. Use whenever editing index.html, adding/fixing a news source, debugging a CORS issue, or shipping a UI change.
---

# Working in ai-news-hub

## When to use

Trigger this skill any time the user:
- Edits `index.html` (the only source file)
- Adds, removes, or fixes a news source
- Reports a card with missing badges / wrong category / wrong source / empty title
- Asks about CORS, rate limits, or proxies
- Touches the filter UI (category tabs, source chips, day window, search, download menu)
- Asks for a build / test / install command (there isn't one — see workflow)

## Repository overview

**One file. No build. No deps. No tests. No CI.** The entire app is `C:\Users\Eddie\workspace\ai-news-hub\index.html` (~1350 lines). It's a static SPA that runs by opening the file in a browser — no `package.json`, no `requirements.txt`, no Makefile, no Docker, no GitHub Actions.

Stack: vanilla HTML + CSS custom properties (dark/light themed via `[data-theme]`) + plain ES2020+ JS. No framework, no bundler, no transpilation. The zero-dependency property is **load-bearing** — don't introduce npm/pip/CDN scripts without an explicit ask.

### Sections inside `index.html`

The JS is divided by banner comments `/* ── Section ── */`. To jump around: `Grep ^/\* ──`. Current section line numbers:

| Line ~ | Section            | Purpose                                                          |
|--------|--------------------|------------------------------------------------------------------|
| 403    | `Sources`          | `SOURCES` config — name, short label, brand color, badge bg      |
| 416    | `Category config`  | `CATS` — category keywords used by `categorize()`                |
| 454    | `Config`           | `REFRESH_MS`, `MIN_LAB_DAYS`, `PRESETS`, hour bounds             |
| 473    | `State`            | `windowHours`, `allStories`, `activeTab`, `activeSource`, etc.   |
| 489    | `Helpers`          | `domain()`, `timeAgo()`, `sinceTs()`, `windowLabel()`, `esc()`   |
| 555    | `Source fetchers`  | One async fn per source (9 total — see table below)              |
| 929    | `Grid`             | `renderGrid()` — produces card HTML                              |
| 1005   | `Meta row`         | `renderMeta()` — window selector + counts                        |
| 1084   | `Main fetch`       | `fetchNews()` — orchestrator (`Promise.allSettled` across sources) |
| 1161   | `Download / export`| `exportJSON()`, `exportCSV()`, `exportMarkdown()`                |
| 1314   | `Events`           | Refresh / theme / search / download menu / badge capture-phase   |

These shift whenever you add a fetcher. Re-run the Grep if line numbers look off.

### Source fetchers

| Fn                | Line | Source         | Strategy                                                  |
|-------------------|------|----------------|-----------------------------------------------------------|
| `fetchHN`         | 556  | HN Algolia     | 6 AI-keyword queries, `search_by_date`                   |
| `fetchDevTo`      | 574  | DEV.to         | Tag filter, `state=fresh`                                |
| `fetchThroughProxy` | 596 | (helper)     | CORS-proxy chain: corsproxy.io → allorigins → codetabs    |
| `fetchArxiv`      | 614  | arXiv Atom     | Through proxy, 48 h window (daily batch cadence)         |
| `fetchReddit`     | 656  | Reddit         | `r/{MachineLearning,artificial,LocalLLaMA}/new.json`     |
| `fetchAnthropic`  | 682  | anthropic.com  | HN Algolia URL search, strict `/research/<slug>` regex   |
| `fetchOpenAI`     | 723  | openai.com     | HN Algolia URL search, `/research/<slug>` + `/index/<slug>` |
| `fetchGoogle`     | 771  | Google         | Two parallel HN searches (research.google + deepmind.google) |
| `fetchX`          | 820  | X / Twitter    | HN Algolia URL search + AI-keyword title gate            |
| `fetchLobsters`   | 851  | Lobste.rs      | `/t/<tag>.json` for `ai`, `ml`                           |

`fetchNews()` runs all 9 in `Promise.allSettled` and concatenates the lab/X feeds first so their URL-tagged versions win the URL-keyed dedup against generic HN/Reddit submissions.

## Build / test / run commands

There are none in the traditional sense.

- **Run / "build":** open `index.html` in a browser. The Launch preview panel does this automatically on every write.
- **Test:** there is no test runner. Manual smoke test via the preview panel. For source-fetch debugging, the user opens DevTools (F12) → Console — every fetcher logs `[SourceName] N stories …` and the proxy logs `[arXiv] all CORS proxies failed` on the fallback path.
- **Lint / typecheck:** none configured.
- **Deploy:** static file — drop `index.html` on any web host or open locally.

## Common workflows

### Add a new news source (canonical recipe)

1. **Register it** in `SOURCES` (~line 403). Pick a brand color and short label. Lab/publisher sources sit at the top of the object — their order is also their dedup-priority order.
2. **Write `fetchXyz(since)`** in `Source fetchers`. Must return an array of records with these exact fields (mirror `fetchAnthropic`):
   ```js
   {
     id, title, url, discussionUrl,
     sourceKey,         // matches the SOURCES key
     points, comments,
     created_at_i,      // Unix seconds — required for recency filter + sort + dedup
     author,
   }
   ```
3. **Wire into `fetchNews()`** (~line 1084): add to the `Promise.allSettled([...])` array, the destructured result variable, the `raw` spread, *and* the `nOk/N sources OK` toast counter.
4. **Update the loading message** `Fetching from N sources…` (in the same `fetchNews()`).
5. **Console-log the count** at the end of your fetcher so the user can verify in DevTools.

### Fix "source shows zero stories"

In order of likelihood:
- **No CORS headers** on the upstream API → route via `fetchThroughProxy()` (chain: corsproxy.io → allorigins.win → codetabs.com). arXiv is the existing example.
- **Source restructured URLs** (e.g. OpenAI moved research from `/research/` to `/index/`) → broaden the Algolia `query` to just the bare domain, then tighten with a regex post-filter on `h.url`. Each branch should require a slug like `\/[a-z0-9-]+` so landing pages don't slip through.
- **Window too narrow** for a low-cadence source → labs (Anthropic, OpenAI, Google) enforce a 14-day floor: `const floor = Math.floor(Date.now()/1000) - MIN_LAB_DAYS*86400; const xSince = Math.min(since, floor);`. New labs should adopt this pattern.
- **Empty / missing title** → already filtered in `fetchNews()`'s dedup step (`if (!s.title.trim()) return false;`). Check the per-source console log to verify the fetcher itself is returning items.

### Test a regex/path filter before shipping

In DevTools console: `/<your-regex>/.test('https://www.research.google/blog/foo')`. Run it against both a positive case (a real article URL) and a negative case (a `/careers` or `/about` page) to confirm both directions work.

### Avoid the wrapping-`<a>` click trap

Cards are an `<a>` around the entire tile so the whole card opens the article. Anything *inside* that should NOT navigate (badges, "HN ↗" link, format buttons in the download menu) must `event.preventDefault()` + `event.stopPropagation()` in the **capture phase**. The badge-filter listener (~line 1316) is the canonical example.

## Coding conventions

- **Edit in place.** Use `Edit`, not `Write`. The single-file structure means a full rewrite drops anchor points the user has been pointing to in earlier turns.
- **Banner comments are the architecture.** `/* ── X ── */` is how sections are organized. New code goes inside an existing banner or under a new one.
- **HTML escaping is mandatory** for any string from a third-party API injected into `innerHTML`. Use `esc()`. Story titles, author names, and URLs are untrusted.
- **No frameworks, no bundler, no JSX.** Plain `innerHTML` template strings + event delegation. Don't introduce React/Vue/anything that needs npm.
- **CSS lives in `<style>`** at the top of the file with the same banner-comment sectioning. Theme colors are CSS custom properties on `:root[data-theme="..."]` — never hardcode `#fff` / `#000`.
- **URL dedup ordering matters.** In `fetchNews()`, source-specific feeds (Anthropic, OpenAI, Google, X) run *before* generic feeds (HN, DEV.to, Reddit, arXiv, Lobste.rs) so their tagged version wins the URL-keyed dedup. Don't reorder casually.
- **State persists to `localStorage`** via `windowHours` (`setWindowHours`). Read state on init, write on change.

## CORS / API gotchas

| API                            | CORS? | Notes                                                                  |
|--------------------------------|-------|------------------------------------------------------------------------|
| HN Algolia (`hn.algolia.com`)  | ✅    | Free, no key. Use `search_by_date` for recency, not `search`.          |
| DEV.to (`dev.to/api`)          | ✅    | Use `tag=` filter + `state=fresh`.                                     |
| Reddit (`reddit.com/*.json`)   | ✅    | `raw_json=1` to get unescaped strings.                                 |
| Lobste.rs                      | ✅    | `/t/<tag>.json`.                                                       |
| arXiv (`export.arxiv.org`)     | ❌    | **Must** go through `fetchThroughProxy()`. Returns Atom XML; parse with `DOMParser` + `getElementsByTagName` (`querySelectorAll` is unreliable with the default namespace). |
| X / Twitter                    | ❌❌  | No public API since 2023. Surface tweets via HN Algolia URL search.    |

## Validation checklist before finishing

- [ ] Used `Edit` (not `Write`) on `index.html`.
- [ ] Any third-party string interpolated into `innerHTML` is wrapped in `esc()`.
- [ ] If a fetcher was added: count appears in the meta-row "M sources", in the source-chip row (renderSourceChips auto-derives from `SOURCES`), and in the `X/N sources OK` toast — plus the loading message reads "Fetching from N sources…".
- [ ] If a regex / path filter was changed: tested both a positive and negative URL in DevTools console.
- [ ] If a badge / dropdown / inline link was added: confirmed it intercepts click in the capture phase so the wrapping `<a>` doesn't navigate.
- [ ] Told the user the change is "live in the preview panel" so they refresh and look.
- [ ] No build tooling, framework, or external script tag introduced without explicit request.
- [ ] Banner-comment section numbers in this skill still match — if you added a fetcher, the line table above is now stale by ~30–60 lines. Re-run `Grep ^/\* ──` and update.

## Windows-specific notes

- Repo lives at `C:\Users\Eddie\workspace\ai-news-hub`. Use backslash paths in `Read` / `Write` / `Edit` calls and quote paths in `Bash` commands.
- `mkdir -p` works under the bundled bash; PowerShell equivalent is `New-Item -ItemType Directory -Force`.
- No build / run command. "Running" the app = opening `index.html` in a browser; the Launch preview panel does this on every write.
