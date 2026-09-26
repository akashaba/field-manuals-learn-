// Extract LeetCode problem slugs from every Destination-FAANG Java file
// and fetch the question content via LeetCode's public GraphQL. Cached in
// content/leetcode/<slug>.json so subsequent builds are instant.

import fs from "node:fs/promises";
import path from "node:path";

const SUBJECT_SLUG = "destination-faang";
const RAW_ROOT = path.resolve("content/raw", SUBJECT_SLUG);
const CACHE_DIR = path.resolve("content/leetcode");
const REQUEST_DELAY_MS = 500; // be polite

const QUERY = `
  query questionData($titleSlug: String!) {
    question(titleSlug: $titleSlug) {
      questionFrontendId
      title
      titleSlug
      difficulty
      content
      exampleTestcases
      topicTags { name slug }
    }
  }
`.trim();

async function walkJava(dir) {
  const out = [];
  let entries;
  try { entries = await fs.readdir(dir, { withFileTypes: true }); }
  catch { return out; }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...await walkJava(p));
    else if (e.name.endsWith(".java")) out.push(p);
  }
  return out;
}

function extractSlug(javaSource) {
  const m = javaSource.match(/Leetcode\s*Link\s*:\s*https?:\/\/leetcode\.com\/problems\/([A-Za-z0-9-]+)/i);
  return m ? m[1] : null;
}

async function fetchQuestion(slug) {
  const body = JSON.stringify({ query: QUERY, variables: { titleSlug: slug } });
  const r = await fetch("https://leetcode.com/graphql", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "User-Agent": "Mozilla/5.0 (learn-AI content builder)",
      "Referer": `https://leetcode.com/problems/${slug}/`,
    },
    body,
  });
  if (!r.ok) throw new Error(`http ${r.status}`);
  const j = await r.json();
  if (j.errors) throw new Error(`graphql: ${JSON.stringify(j.errors)}`);
  if (!j.data?.question) throw new Error("no question in response");
  return j.data.question;
}

async function main() {
  await fs.mkdir(CACHE_DIR, { recursive: true });

  const javaFiles = await walkJava(RAW_ROOT);
  if (javaFiles.length === 0) {
    console.log(`[leetcode] no destination-faang java files under ${RAW_ROOT} — skipping.`);
    return;
  }

  const slugs = new Set();
  for (const f of javaFiles) {
    const src = await fs.readFile(f, "utf8");
    const slug = extractSlug(src);
    if (slug) slugs.add(slug);
  }
  console.log(`[leetcode] found ${slugs.size} unique problem slugs across ${javaFiles.length} java files.`);

  let fetched = 0, cached = 0, failed = 0;
  for (const slug of slugs) {
    const target = path.join(CACHE_DIR, `${slug}.json`);
    try { await fs.access(target); cached++; continue; } catch {}
    try {
      const data = await fetchQuestion(slug);
      await fs.writeFile(target, JSON.stringify(data, null, 2), "utf8");
      fetched++;
      if (fetched % 10 === 0) console.log(`[leetcode]   ...${fetched} fetched`);
      await new Promise((r) => setTimeout(r, REQUEST_DELAY_MS));
    } catch (e) {
      failed++;
      console.warn(`[leetcode]   ! ${slug}: ${e.message}`);
      // negative-cache marker so we don't retry every build
      await fs.writeFile(target, JSON.stringify({ __error: e.message, slug }, null, 2), "utf8");
    }
  }

  console.log(`[leetcode] done. fetched=${fetched} cached=${cached} failed=${failed}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
