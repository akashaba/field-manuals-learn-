// Fetch every subject's lesson content into content/raw/<subject-slug>/...
// Idempotent, cache-aware — skips files whose sha matches what we already have.

import fs from "node:fs/promises";
import path from "node:path";
import { SUBJECTS } from "../lib/subjects.js";

const RAW_ROOT = path.resolve("content/raw");
const SHA_INDEX = path.resolve("content/shas.json");
const CONCURRENCY = 8;

async function loadShas() {
  try { return JSON.parse(await fs.readFile(SHA_INDEX, "utf8")); }
  catch { return {}; }
}

async function pool(items, n, fn) {
  let i = 0;
  const workers = Array.from({ length: n }, async () => {
    while (i < items.length) {
      const idx = i++;
      await fn(items[idx], idx);
    }
  });
  await Promise.all(workers);
}

async function fetchSubject(subject, shas) {
  const { owner, name, branch } = subject.repo;
  console.log(`\n[${subject.slug}] Fetching git tree for ${owner}/${name}@${branch} ...`);

  const treeResp = await fetch(
    `https://api.github.com/repos/${owner}/${name}/git/trees/${branch}?recursive=1`,
    { headers: { "User-Agent": "learn-AI-generator" } },
  );
  if (!treeResp.ok) throw new Error(`tree fetch failed for ${subject.slug}: ${treeResp.status}`);
  const tree = await treeResp.json();
  if (!tree.tree) throw new Error(`empty tree for ${subject.slug}`);

  const candidates = tree.tree
    .filter((t) => t.type === "blob")
    .map((t) => ({ sha: t.sha, path: t.path, matched: subject.matcher(t.path) }))
    .filter((t) => t.matched);

  console.log(`[${subject.slug}] Found ${candidates.length} lesson docs.`);

  let fetched = 0, skipped = 0, failed = 0;
  await pool(candidates, CONCURRENCY, async (entry) => {
    const key = `${subject.slug}:${entry.path}`;
    const target = path.join(RAW_ROOT, subject.slug, entry.path);
    if (shas[key] === entry.sha) {
      try { await fs.access(target); skipped++; return; } catch {}
    }
    try {
      const rawUrl = `https://raw.githubusercontent.com/${owner}/${name}/${branch}/${entry.path.split("/").map(encodeURIComponent).join("/")}`;
      const r = await fetch(rawUrl);
      if (!r.ok) throw new Error(`${r.status} ${entry.path}`);
      const body = await r.text();
      await fs.mkdir(path.dirname(target), { recursive: true });
      await fs.writeFile(target, body, "utf8");
      shas[key] = entry.sha;
      fetched++;
      if (fetched % 25 === 0) console.log(`[${subject.slug}]   ...${fetched} fetched`);
    } catch (e) {
      failed++;
      console.warn(`[${subject.slug}]   ! ${entry.path}: ${e.message}`);
    }
  });

  console.log(`[${subject.slug}] Done. fetched=${fetched} skipped=${skipped} failed=${failed}`);
}

async function main() {
  await fs.mkdir(RAW_ROOT, { recursive: true });
  const shas = await loadShas();
  for (const subject of SUBJECTS) {
    if (subject.source === "local-pdf") {
      console.log(`\n[${subject.slug}] local-pdf — skipping GitHub fetch (use \`npm run content:pdf\`)`);
      continue;
    }
    if (subject.source === "articles-index") {
      console.log(`\n[${subject.slug}] articles-index — skipping GitHub tree fetch (use \`npm run content:articles\`)`);
      continue;
    }
    await fetchSubject(subject, shas);
  }
  await fs.mkdir(path.dirname(SHA_INDEX), { recursive: true });
  await fs.writeFile(SHA_INDEX, JSON.stringify(shas, null, 2), "utf8");
  console.log("\nAll subjects fetched.");
}

main().catch((e) => { console.error(e); process.exit(1); });
