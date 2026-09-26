// Import the harshit3011/Technical-Engineering-Articles article index.
// Reads articles.json from the repo, groups by category, and emits one
// markdown file per article that renders with X's official tweet embed.

import fs from "node:fs/promises";
import path from "node:path";

const OWNER = "harshit3011";
const REPO = "Technical-Engineering-Articles";
const BRANCH = "main";
const SUBJECT_SLUG = "tech-articles";
const RAW_ROOT = path.resolve("content/raw", SUBJECT_SLUG);

// category name → phase slug
const CATEGORY_TO_SLUG = {
  "HLD": "hld",
  "LLD": "lld",
  "Backend Engineering": "backend-engineering",
  "Engineering Articles": "engineering-articles",
  "Distributed Systems": "distributed-systems",
  "Microservices": "microservices",
  "Machine Learning": "machine-learning",
  "AI Engineering": "ai-engineering",
};

function slugify(s) {
  return String(s).toLowerCase().trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}
function pad2(n) { return String(n).padStart(2, "0"); }

// Escape < > & so raw HTML inside markdown doesn't break marked's parser.
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function authorFromUrl(url) {
  try {
    const u = new URL(url);
    const m = u.pathname.match(/^\/([^/]+)\/status\//);
    return m ? m[1] : null;
  } catch { return null; }
}

async function main() {
  const url = `https://raw.githubusercontent.com/${OWNER}/${REPO}/${BRANCH}/articles.json`;
  console.log(`[articles] fetching ${url}`);
  const raw = await fetch(url).then((r) => r.json());
  const articles = Array.isArray(raw)
    ? raw
    : Array.isArray(raw.articles) ? raw.articles
    : Array.isArray(raw.items) ? raw.items
    : Object.values(raw).find(Array.isArray);
  if (!Array.isArray(articles)) throw new Error("could not find an array of articles in articles.json");

  console.log(`[articles] ${articles.length} entries`);

  // Group and sort by date within each group
  const byCategory = new Map();
  for (const a of articles) {
    const catSlug = CATEGORY_TO_SLUG[a.category] || slugify(a.category);
    if (!byCategory.has(catSlug)) byCategory.set(catSlug, []);
    byCategory.get(catSlug).push(a);
  }
  for (const arr of byCategory.values()) {
    arr.sort((a, b) => String(a.date).localeCompare(String(b.date)));
  }

  await fs.rm(RAW_ROOT, { recursive: true, force: true });
  await fs.mkdir(RAW_ROOT, { recursive: true });

  let total = 0;
  for (const [catSlug, items] of byCategory) {
    const dir = path.join(RAW_ROOT, catSlug);
    await fs.mkdir(dir, { recursive: true });

    for (let i = 0; i < items.length; i++) {
      const a = items[i];
      const author = authorFromUrl(a.url);
      const slug = `${pad2(i + 1)}-${slugify(a.title)}`;
      const md = `# ${a.title}

> ${a.description}

**Category:** ${a.category}
**Date:** ${a.date}
**Author:** [@${author || "author"}](https://x.com/${author || ""})
**Source:** [Original thread on X](${a.url})

<blockquote class="twitter-tweet" data-conversation="none" data-dnt="true">
  <p lang="en" dir="ltr">Loading thread from X&hellip;</p>
  &mdash; <a href="${escapeHtml(a.url)}">${escapeHtml(a.title)}</a>
</blockquote>

The thread renders live from X above. If it doesn't load — X sometimes blocks embeds for signed-out viewers — the [full thread is here](${a.url}).
`;
      await fs.writeFile(path.join(dir, `${slug}.md`), md, "utf8");
      total++;
    }
    console.log(`[articles] ${catSlug}: ${items.length} articles`);
  }
  console.log(`[articles] wrote ${total} article pages to ${RAW_ROOT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
