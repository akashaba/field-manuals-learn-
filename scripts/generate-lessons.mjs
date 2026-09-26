// Read every content/raw/<subject>/... lesson file and emit
// public/lessons/<subject>/<group>/<item>.html + content/manifest.json.

import fs from "node:fs/promises";
import path from "node:path";
import { marked } from "marked";
import { SUBJECTS } from "../lib/subjects.js";

const RAW_ROOT = path.resolve("content/raw");
const OUT_ROOT = path.resolve("public/lessons");
const MANIFEST = path.resolve("content/manifest.json");

// Lessons rewritten by hand — the generator preserves their existing HTML.
// Keys are "<subject-slug>/<group>/<item>".
const FLAGSHIP = new Set([
  "ai-engineering/10-llms-from-scratch/07-rlhf",
]);

// ---------- markdown pipeline ---------------------------------------------
function makeRenderer(subject, imageBase) {
  const renderer = new marked.Renderer();

  renderer.code = function ({ text, lang }) {
    const language = (lang || "").trim().toLowerCase();
    if (language === "mermaid") {
      return `<div class="fig"><pre class="mermaid">${escapeHtml(text)}</pre></div>`;
    }
    const cls = language ? `language-${language} hljs` : "hljs";
    const label = language ? `<span class="cb-label">${language}</span>` : "";
    return `
      <div class="codeblock" data-lang="${escapeAttr(language)}">
        ${label}
        <button class="cb-copy" type="button" aria-label="Copy code">copy</button>
        <pre><code class="${cls}">${escapeHtml(text)}</code></pre>
      </div>
    `;
  };

  renderer.heading = function ({ tokens, depth }) {
    const text = this.parser.parseInline(tokens);
    const raw  = tokens.map(t => t.raw || t.text || "").join("");
    const id = slugify(raw);
    return `<h${depth} id="${id}"><a class="anchor" href="#${id}">#</a> ${text}</h${depth}>\n`;
  };

  // Rewrite relative image URLs. For GitHub-sourced subjects, resolve to
  // raw.githubusercontent.com. For local-pdf subjects, images live under
  // public/lessons/<slug>/chapters/... — serve them from that path.
  const { owner, name, branch } = subject.repo;
  const isLocalPdf = subject.source === "local-pdf";
  renderer.image = function ({ href, title, text }) {
    let src = href || "";
    if (!/^https?:|^data:|^\/\//i.test(src)) {
      const cleaned = src.replace(/^\.\/+/, "");
      const parts = (imageBase + cleaned).split("/").filter(Boolean);
      const stack = [];
      for (const p of parts) {
        if (p === "..") stack.pop();
        else if (p !== ".") stack.push(p);
      }
      const resolved = stack.map(encodeURIComponent).join("/");
      src = isLocalPdf
        ? `/lessons/${subject.slug}/${resolved}`
        : `https://raw.githubusercontent.com/${owner}/${name}/${branch}/${resolved}`;
    }
    const titleAttr = title ? ` title="${escapeAttr(title)}"` : "";
    return `<figure class="ext-image"><img src="${escapeAttr(src)}" alt="${escapeAttr(text || "")}"${titleAttr} loading="lazy" /></figure>`;
  };

  return renderer;
}

// Rewrite raw <img src="..."> attributes that use relative paths so they hit
// raw.githubusercontent.com. Runs after marked, so it catches HTML img tags
// the source markdown embeds directly.
function rewriteRawImageSrcs(html, subject, imageBase) {
  const { owner, name, branch } = subject.repo;
  const base = imageBase;
  const isLocalPdf = subject.source === "local-pdf";
  return html.replace(/<img\b([^>]*)\bsrc=(["'])([^"']+)\2/gi, (m, prefix, q, src) => {
    // Skip anything already resolved: absolute URLs, data URIs, and any path
    // that starts with "/" (we already rewrote it via the marked renderer).
    if (/^https?:|^data:|^\//i.test(src)) return m;
    const cleaned = src.replace(/^\.\/+/, "");
    const parts = (base + cleaned).split("/").filter(Boolean);
    const stack = [];
    for (const p of parts) {
      if (p === "..") stack.pop();
      else if (p !== ".") stack.push(p);
    }
    const resolved = stack.map(encodeURIComponent).join("/");
    const abs = isLocalPdf
      ? `/lessons/${subject.slug}/${resolved}`
      : `https://raw.githubusercontent.com/${owner}/${name}/${branch}/${resolved}`;
    return `<img${prefix}src=${q}${abs}${q}`;
  });
}

function escapeHtml(s = "") {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function escapeAttr(s = "") { return escapeHtml(s); }
function slugify(s) {
  return s.toLowerCase().trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .slice(0, 80);
}

// ---------- YouTube embed post-processor ----------------------------------
function youtubeId(url) {
  try {
    const u = new URL(url);
    if (/(^|\.)youtu\.be$/i.test(u.hostname)) return u.pathname.slice(1).split("/")[0] || null;
    if (/(^|\.)youtube\.com$/i.test(u.hostname)) {
      if (u.pathname === "/watch") return u.searchParams.get("v");
      const em = u.pathname.match(/^\/embed\/([^/?]+)/);
      if (em) return em[1];
      const sh = u.pathname.match(/^\/shorts\/([^/?]+)/);
      if (sh) return sh[1];
    }
  } catch {}
  return null;
}
function embedYouTube(html) {
  // Replace <a href="youtube link">…</a> anchors with responsive iframe embeds.
  return html.replace(/<a\b([^>]*?)href=(["'])([^"']+)\2([^>]*)>([\s\S]*?)<\/a>/gi, (m, pre, q, href, post, inner) => {
    const id = youtubeId(href);
    if (!id) return m;
    return `<figure class="yt-embed">
      <div class="yt-frame"><iframe src="https://www.youtube.com/embed/${encodeURIComponent(id)}"
        title="YouTube video" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        referrerpolicy="strict-origin-when-cross-origin" allowfullscreen loading="lazy"></iframe></div>
    </figure>`;
  });
}

// ---------- notebook (.ipynb) renderer ------------------------------------
function stripScripts(html) {
  return html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "");
}

function renderNotebookOutputs(outputs) {
  if (!outputs || !outputs.length) return "";
  const chunks = [];
  for (const o of outputs) {
    if (o.output_type === "stream") {
      const text = Array.isArray(o.text) ? o.text.join("") : (o.text || "");
      chunks.push(`<pre class="nb-stream"><code>${escapeHtml(text)}</code></pre>`);
    } else if (o.output_type === "error") {
      const trace = Array.isArray(o.traceback) ? o.traceback.join("\n") : (o.traceback || `${o.ename}: ${o.evalue}`);
      const clean = trace.replace(/\[[0-9;]*m/g, "");
      chunks.push(`<pre class="nb-err"><code>${escapeHtml(clean)}</code></pre>`);
    } else if (o.output_type === "execute_result" || o.output_type === "display_data") {
      const data = o.data || {};
      if (data["image/png"]) {
        const b64 = Array.isArray(data["image/png"]) ? data["image/png"].join("") : data["image/png"];
        chunks.push(`<figure class="nb-image"><img alt="notebook output" src="data:image/png;base64,${b64.replace(/\s+/g, "")}"/></figure>`);
      } else if (data["image/jpeg"]) {
        const b64 = Array.isArray(data["image/jpeg"]) ? data["image/jpeg"].join("") : data["image/jpeg"];
        chunks.push(`<figure class="nb-image"><img alt="notebook output" src="data:image/jpeg;base64,${b64.replace(/\s+/g, "")}"/></figure>`);
      } else if (data["image/svg+xml"]) {
        const svg = Array.isArray(data["image/svg+xml"]) ? data["image/svg+xml"].join("") : data["image/svg+xml"];
        chunks.push(`<figure class="nb-image">${stripScripts(svg)}</figure>`);
      } else if (data["text/html"]) {
        const raw = Array.isArray(data["text/html"]) ? data["text/html"].join("") : data["text/html"];
        chunks.push(`<div class="nb-html">${stripScripts(raw)}</div>`);
      } else if (data["text/plain"]) {
        const raw = Array.isArray(data["text/plain"]) ? data["text/plain"].join("") : data["text/plain"];
        chunks.push(`<pre class="nb-out"><code>${escapeHtml(raw)}</code></pre>`);
      }
    }
  }
  return chunks.join("\n");
}

function renderNotebook(nb, subject, imageBase) {
  const language = (nb.metadata?.kernelspec?.language || nb.metadata?.language_info?.name || "python").toLowerCase();
  const renderer = makeRenderer(subject, imageBase);
  marked.use({ renderer, gfm: true });

  const cells = nb.cells || [];
  const bodyParts = [];
  let title = "";
  let subtitle = "";
  const toc = [];

  for (const cell of cells) {
    const src = Array.isArray(cell.source) ? cell.source.join("") : (cell.source || "");
    if (cell.cell_type === "markdown") {
      if (!title) {
        const h1 = src.match(/^\s*#\s+(.+?)\s*$/m);
        if (h1) title = h1[1].trim();
      }
      if (title && !subtitle) {
        const bq = src.match(/^\s*>\s+(.+?)\s*$/m);
        if (bq) subtitle = bq[1].trim();
      }
      // pick up h2 headings for TOC
      const h2s = [...src.matchAll(/^##\s+(.+)$/gm)];
      for (const m of h2s) {
        const text = m[1].trim();
        toc.push({ text, id: slugify(text) });
      }
      const html = marked.parse(src);
      bodyParts.push(`<section class="nb-md">${html}</section>`);
    } else if (cell.cell_type === "code") {
      const label = language ? `<span class="cb-label">${escapeHtml(language)}</span>` : "";
      bodyParts.push(`
        <div class="codeblock" data-lang="${escapeAttr(language)}">
          ${label}
          <button class="cb-copy" type="button" aria-label="Copy code">copy</button>
          <pre><code class="language-${escapeAttr(language)} hljs">${escapeHtml(src)}</code></pre>
        </div>
        ${renderNotebookOutputs(cell.outputs)}
      `);
    } else if (cell.cell_type === "raw") {
      bodyParts.push(`<pre class="nb-raw"><code>${escapeHtml(src)}</code></pre>`);
    }
  }

  return {
    title: title || null,
    subtitle: subtitle || null,
    meta: {},
    body: "",
    toc,
    prerendered: bodyParts.join("\n"),
  };
}

// ---------- Java LeetCode-solution renderer -------------------------------
// Destination-FAANG's files contain: header comments with a Leetcode link and
// a YouTube video, then a "*** Java Solution ***" divider, then the code.
// We render: problem statement (pulled from LeetCode) + video embed + code.
function parseJavaLesson(src) {
  const leet = src.match(/Leetcode\s*Link\s*:\s*(https?:\/\/leetcode\.com\/problems\/([A-Za-z0-9-]+)\/?)/i);
  const yt   = src.match(/Video\s*Solution\s*:\s*(https?:\/\/[^\s"]+)/i);
  const dividerIdx = src.search(/\*{4,}\s*Java\s*Solution\s*\*{4,}/i);
  const code = dividerIdx >= 0
    ? src.slice(dividerIdx).replace(/\*{4,}\s*Java\s*Solution\s*\*{4,}\n?/i, "").trim()
    // fallback: everything from the first "class " or "public " onward
    : src.slice(Math.max(0, src.search(/^\s*(public\s+)?class\s+/m))).trim();
  return {
    leetcodeUrl: leet?.[1] || null,
    leetcodeSlug: leet?.[2] || null,
    videoUrl: yt?.[1] || null,
    code: code || src,
  };
}

async function loadLeetCodeQuestion(slug) {
  if (!slug) return null;
  const p = path.resolve("content/leetcode", `${slug}.json`);
  try {
    const j = JSON.parse(await fs.readFile(p, "utf8"));
    if (j.__error) return null;
    return j;
  } catch { return null; }
}

function renderJavaLesson({ folderTitle, parsed, question }) {
  const parts = [];
  const meta = {};

  // meta strip content (Difficulty, ID, Tags)
  if (question?.difficulty) meta["Difficulty"] = question.difficulty;
  if (question?.questionFrontendId) meta["LC #"] = question.questionFrontendId;
  if (question?.topicTags?.length) meta["Tags"] = question.topicTags.map(t => t.name).slice(0, 4).join(", ");

  const displayTitle = question?.title || folderTitle.replace(/^\d+\s+/, "");
  const subtitle = question?.difficulty
    ? `LeetCode #${question.questionFrontendId} · ${question.difficulty}`
    : null;

  // Problem statement (from LeetCode HTML)
  if (question?.content) {
    parts.push(`
      <h2 id="problem"><a class="anchor" href="#problem">#</a> Problem statement</h2>
      <div class="leetcode-content">${stripScripts(question.content)}</div>
    `);
    if (question.exampleTestcases) {
      parts.push(`
        <h3>Example test cases</h3>
        <div class="codeblock" data-lang="text">
          <span class="cb-label">test cases</span>
          <button class="cb-copy" type="button" aria-label="Copy code">copy</button>
          <pre><code class="hljs">${escapeHtml(question.exampleTestcases)}</code></pre>
        </div>
      `);
    }
  } else if (parsed.leetcodeUrl) {
    parts.push(`
      <div class="callout">
        <span class="tag">Problem</span>
        LeetCode question content couldn't be pulled at build time.
        <a href="${escapeAttr(parsed.leetcodeUrl)}" target="_blank" rel="noreferrer">Open on LeetCode ↗</a>
      </div>
    `);
  }

  // Video embed — reuse embedYouTube by feeding it an <a>
  if (parsed.videoUrl) {
    const anchor = `<a href="${escapeAttr(parsed.videoUrl)}">video</a>`;
    const embedded = embedYouTube(anchor);
    parts.push(`
      <h2 id="video-solution"><a class="anchor" href="#video-solution">#</a> Video walkthrough</h2>
      ${embedded === anchor ? `<p><a href="${escapeAttr(parsed.videoUrl)}" target="_blank" rel="noreferrer">${escapeHtml(parsed.videoUrl)}</a></p>` : embedded}
    `);
  }

  // Java solution
  parts.push(`
    <h2 id="java-solution"><a class="anchor" href="#java-solution">#</a> Java solution</h2>
    <div class="codeblock" data-lang="java">
      <span class="cb-label">java</span>
      <button class="cb-copy" type="button" aria-label="Copy code">copy</button>
      <pre><code class="language-java hljs">${escapeHtml(parsed.code)}</code></pre>
    </div>
  `);

  // Footer link back to LeetCode
  if (parsed.leetcodeUrl) {
    parts.push(`<p class="lc-link"><a href="${escapeAttr(parsed.leetcodeUrl)}" target="_blank" rel="noreferrer">Solve this on LeetCode ↗</a></p>`);
  }

  const toc = [
    ...(question?.content ? [{ text: "Problem statement", id: "problem" }] : []),
    ...(parsed.videoUrl ? [{ text: "Video walkthrough", id: "video-solution" }] : []),
    { text: "Java solution", id: "java-solution" },
  ];

  return {
    title: displayTitle,
    subtitle,
    meta,
    toc,
    htmlBody: parts.join("\n"),
  };
}

// ---------- lesson metadata parsing ---------------------------------------
function parseLesson(md) {
  const lines = md.split("\n");
  let title = "";
  let subtitle = "";
  const meta = {};
  const bodyLines = [];
  let inPrelude = true;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!title && line.startsWith("# ")) { title = line.slice(2).trim(); continue; }
    if (inPrelude && title && !subtitle && line.startsWith("> ")) {
      subtitle = line.slice(2).trim();
      continue;
    }
    if (inPrelude) {
      const kv = line.match(/^\*\*([^*:]+):\*\*\s*(.+)$/);
      if (kv) { meta[kv[1].trim()] = kv[2].trim(); continue; }
    }
    // once we hit substantive content, stop consuming metadata lines
    if (line.trim() !== "") inPrelude = false;
    bodyLines.push(line);
  }

  const body = bodyLines.join("\n");
  const tocMatches = [...body.matchAll(/^##\s+(.+)$/gm)].map((m) => ({
    text: m[1].trim(),
    id: slugify(m[1].trim()),
  }));
  return { title, subtitle, meta, body, toc: tocMatches };
}

// ---------- per-lesson sigil ----------------------------------------------
function hashStr(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
  return h;
}
function renderSigil({ subject, groupLabel, itemNum, itemTitle }) {
  const W = 320, H = 180;
  const seed = hashStr(`${subject.slug}/${groupLabel}/${itemNum}/${itemTitle}`);
  const rand = (n) => { let x = (seed + n * 2654435761) >>> 0; return (x & 0xffff) / 0xffff; };
  const palette = [subject.accent, subject.accent2, "#a97817", "#37613b", "#8f2f20", "#55503f"];
  const acc  = palette[Math.floor(rand(1) * palette.length)];
  const acc2 = palette[(Math.floor(rand(2) * palette.length) + 1) % palette.length];

  // Constrain decorative shapes to the LEFT column (x: 8-132) below the top
  // subject strip. Skip anything too close to the numeral so they read as
  // wallpaper, not clutter over the digit.
  const shapes = [];
  const targetCount = 3 + Math.floor(rand(3) * 4);
  let attempts = 0;
  while (shapes.length < targetCount && attempts < targetCount * 6) {
    const i = shapes.length + attempts;
    attempts++;
    const kind = Math.floor(rand(10 + i) * 3);
    const cx = 15 + rand(20 + i) * 110;
    const cy = 45 + rand(30 + i) * 100;
    if (Math.hypot(cx - 70, cy - 100) < 22) continue; // dead-zone around numeral
    const size = 10 + rand(40 + i) * 26;
    const rot  = Math.floor(rand(50 + i) * 360);
    const stroke = shapes.length % 2 === 0 ? acc : acc2;
    if (kind === 0)      shapes.push(`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${(size/2).toFixed(1)}" fill="none" stroke="${stroke}" stroke-width="1.1" opacity="0.55"/>`);
    else if (kind === 1) shapes.push(`<rect x="${(cx-size/2).toFixed(1)}" y="${(cy-size/2).toFixed(1)}" width="${size.toFixed(1)}" height="${size.toFixed(1)}" rx="2" fill="none" stroke="${stroke}" stroke-width="1.1" opacity="0.55" transform="rotate(${rot} ${cx.toFixed(1)} ${cy.toFixed(1)})"/>`);
    else {
      const dx = size / 2;
      shapes.push(`<path d="M${(cx-dx).toFixed(1)},${(cy+dx*0.6).toFixed(1)} L${cx.toFixed(1)},${(cy-dx*0.9).toFixed(1)} L${(cx+dx).toFixed(1)},${(cy+dx*0.6).toFixed(1)} Z" fill="none" stroke="${stroke}" stroke-width="1.1" opacity="0.55"/>`);
    }
  }

  // Wrap the title into up to two ~20-char lines. Ellipsis if it still overflows.
  const words = String(itemTitle).split(/\s+/).filter(Boolean);
  const lines = [];
  let cur = "";
  for (const w of words) {
    const next = cur ? cur + " " + w : w;
    if (next.length <= 20) { cur = next; continue; }
    if (cur) lines.push(cur);
    cur = w;
    if (lines.length >= 2) break;
  }
  if (cur && lines.length < 2) lines.push(cur);
  const displayed = lines.slice(0, 2);
  const shownWordCount = displayed.join(" ").split(/\s+/).filter(Boolean).length;
  if (shownWordCount < words.length && displayed.length) {
    displayed[displayed.length - 1] = displayed[displayed.length - 1].replace(/[.,;:]+$/, "") + "…";
  }
  const titleY = displayed.length === 2 ? 78 : 90;
  const titleTspans = displayed.map((line, i) =>
    `<tspan x="230" dy="${i === 0 ? 0 : 22}" textLength="150" lengthAdjust="spacingAndGlyphs">${escapeHtml(line)}</tspan>`
  ).join("");

  const subjectStr = subject.name.toUpperCase();
  // textLength squashes long subject names to fit; short ones sit at natural width
  const subjectTextLen = Math.min(W - 40, Math.max(80, subjectStr.length * 8));
  const groupTextLen   = Math.min(120, Math.max(40, groupLabel.length * 8));

  return `
    <figure class="sigil">
      <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${escapeAttr(subject.name)} — ${escapeAttr(groupLabel)} ${escapeAttr(itemNum)}">
        <rect width="${W}" height="${H}" fill="#ebe4d5"/>
        ${shapes.join("\n        ")}
        <text x="${W/2}" y="20" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="10" letter-spacing="1.4" fill="${acc}" textLength="${subjectTextLen}" lengthAdjust="spacingAndGlyphs">${escapeHtml(subjectStr)}</text>
        <circle cx="70" cy="100" r="44" fill="none" stroke="${acc}" stroke-width="1" stroke-dasharray="2 4"/>
        <text x="70" y="114" text-anchor="middle" font-family="Fraunces, Georgia, serif" font-weight="900" font-size="48" fill="${acc}" letter-spacing="-2">${escapeHtml(itemNum)}</text>
        <text x="70" y="160" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9" letter-spacing="1.2" fill="#55503f" textLength="${groupTextLen}" lengthAdjust="spacingAndGlyphs">${escapeHtml(groupLabel.toUpperCase())}</text>
        <line x1="148" y1="44" x2="148" y2="156" stroke="#d6ceba"/>
        <text y="${titleY}" text-anchor="middle" font-family="Fraunces, Georgia, serif" font-style="italic" font-size="16" fill="#1c1a17">${titleTspans}</text>
        <text x="230" y="158" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9" letter-spacing="0.6" fill="#55503f">field manual</text>
      </svg>
    </figure>
  `;
}

// ---------- lesson page template ------------------------------------------
function renderPage({ subject, group, groupLabel, item, itemNum, parsed, htmlBody, prev, next, sourceUrl }) {
  const sigil = renderSigil({ subject, groupLabel, itemNum, itemTitle: parsed.title || item });
  const metaList = Object.entries(parsed.meta || {})
    .map(([k, v]) => `<div><span>${escapeHtml(k)}</span><b>${escapeHtml(v)}</b></div>`)
    .join("");
  const tocHtml = parsed.toc
    .map((t) => `<li><a href="#${t.id}">${escapeHtml(t.text)}</a></li>`)
    .join("");
  const title = parsed.title || item;
  const subjectHome = `/${subject.slug}`;
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${escapeHtml(title)} — ${escapeHtml(subject.name)}</title>
<link rel="stylesheet" href="/lesson.css" />
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css" />
<style>:root{--accent:${subject.accent};--accent-2:${subject.accent2};}</style>
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="topbar">
  <a class="home" href="/">Field manuals</a>
  <span class="topbar-sep">›</span>
  <a class="home" href="${subjectHome}"><b>${escapeHtml(subject.name)}</b></a>
  <nav>
    <a href="${subjectHome}">Index</a>
    <a href="${escapeAttr(sourceUrl)}" target="_blank" rel="noreferrer">Source</a>
  </nav>
</header>
<div class="page">
  <aside class="sidenav">
    <div class="crumbs">
      <a href="/">Home</a>
      <span>›</span>
      <a href="${subjectHome}">${escapeHtml(subject.name)}</a>
    </div>
    <div class="phase-label">${escapeHtml(groupLabel)}</div>
    <ol class="toc">${tocHtml}</ol>
  </aside>

  <article id="content" class="prose">
    <div class="eyebrow">${escapeHtml(subject.name)} · ${escapeHtml(groupLabel)}</div>
    <h1 class="lesson-title">${escapeHtml(parsed.title || item)}</h1>
    ${parsed.subtitle ? `<p class="lesson-sub">${escapeHtml(parsed.subtitle)}</p>` : ""}
    ${metaList ? `<div class="meta-strip">${metaList}</div>` : ""}
    ${sigil}
    ${htmlBody}

    <nav class="paginate">
      ${prev ? `<a class="prev" href="${prev.href}"><span>← Previous</span><b>${escapeHtml(prev.title)}</b></a>` : "<span></span>"}
      ${next ? `<a class="next" href="${next.href}"><span>Next →</span><b>${escapeHtml(next.title)}</b></a>` : "<span></span>"}
    </nav>
  </article>
</div>

<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({
    startOnLoad: true, theme: "base", fontFamily: "JetBrains Mono, monospace",
    themeVariables: {
      background: "#ebe4d5", primaryColor: "#f2ede4", primaryTextColor: "#1c1a17",
      primaryBorderColor: "${subject.accent}", secondaryColor: "#faf1de",
      secondaryBorderColor: "#a97817", tertiaryColor: "#eaf1e9", tertiaryBorderColor: "#37613b",
      lineColor: "#55503f", textColor: "#1c1a17", mainBkg: "#f2ede4",
      nodeBorder: "${subject.accent}", clusterBkg: "#ebe4d5", clusterBorder: "#c4baa1",
      titleColor: "#1c1a17", edgeLabelBackground: "#ebe4d5", fontSize: "13px",
    },
    flowchart: { curve: "basis", htmlLabels: true, padding: 12 },
  });
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/typescript.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/rust.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/go.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/java.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/bash.min.js"></script>
${subject.medium === "articles" ? `<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>` : ""}
<script>
  hljs.highlightAll();
  document.querySelectorAll(".cb-copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const code = btn.parentElement.querySelector("code")?.innerText || "";
      try { await navigator.clipboard.writeText(code); btn.textContent = "copied"; }
      catch { btn.textContent = "err"; }
      setTimeout(() => (btn.textContent = "copy"), 1600);
    });
  });
</script>
</body>
</html>`;
}

function renderGroupIndex({ subject, groupLabel, groupSlug, lessons }) {
  const rows = lessons.map((l) => `
    <a class="lesson" href="./${l.slug}.html">
      <span class="lnum">${escapeHtml(l.num)}</span>
      <span class="lname">${escapeHtml(l.title)}${l.flagship ? ' <span class="ribbon">FIELD MANUAL</span>' : ""}</span>
    </a>`).join("");
  return `<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>${escapeHtml(groupLabel)} — ${escapeHtml(subject.name)}</title>
<link rel="stylesheet" href="/lesson.css"/>
<style>:root{--accent:${subject.accent};--accent-2:${subject.accent2};} .ribbon { display:inline-block; font-family:var(--mono); font-size:9px; letter-spacing:.16em; padding:2px 5px; background:var(--accent); color:var(--paper); border-radius:2px; margin-left:8px; vertical-align:middle; }</style>
</head><body>
<header class="topbar">
  <a class="home" href="/">Field manuals</a>
  <span class="topbar-sep">›</span>
  <a class="home" href="/${subject.slug}"><b>${escapeHtml(subject.name)}</b></a>
</header>
<div class="page single">
  <article class="prose">
    <div class="eyebrow">${escapeHtml(subject.name)}</div>
    <h1 class="lesson-title">${escapeHtml(groupLabel)}</h1>
    <div class="phase-lessons">${rows}</div>
  </article>
</div>
</body></html>`;
}

// ---------- walker --------------------------------------------------------
async function walkSubject(subject) {
  const subjectRoot = path.join(RAW_ROOT, subject.slug);
  try { await fs.access(subjectRoot); }
  catch { console.warn(`skip ${subject.slug} — no fetched content`); return []; }

  const out = [];
  async function walk(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const e of entries) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) { await walk(p); continue; }
      const relPath = path.relative(subjectRoot, p).replace(/\\/g, "/");
      const matched = subject.matcher(relPath);
      if (matched) out.push({ ...matched, absPath: p });
    }
  }
  await walk(subjectRoot);
  return out;
}

// ---------- main ----------------------------------------------------------
async function main() {
  await fs.mkdir(OUT_ROOT, { recursive: true });
  const manifest = { subjects: {}, builtAt: new Date().toISOString() };
  let total = 0;

  for (const subject of SUBJECTS) {
    console.log(`\n[${subject.slug}] Rendering ...`);
    const items = await walkSubject(subject);
    if (items.length === 0) continue;

    // Merge multi-source lessons (e.g. rag-from-scratch's CONCEPT.md + CODE.md).
    // Files with the same (group, item) become one lesson entry with a `sources`
    // array. Any source without an explicit role counts as "main".
    const lessonMap = new Map();
    for (const it of items) {
      const key = `${it.group}/${it.item}`;
      if (!lessonMap.has(key)) {
        lessonMap.set(key, {
          group: it.group,
          item: it.item,
          imageBase: it.imageBase,
          isNotebook: !!it.isNotebook,
          sources: [],
        });
      }
      const entry = lessonMap.get(key);
      entry.sources.push({
        role: it.role || "main",
        absPath: it.absPath,
        sourcePath: it.sourcePath,
      });
      if ((it.role || "main") === "main") {
        entry.imageBase = it.imageBase;
        entry.isNotebook = !!it.isNotebook;
        entry.sourcePath = it.sourcePath;
      }
      // fall back for lessons that only ship a companion (rare)
      if (!entry.sourcePath) entry.sourcePath = it.sourcePath;
    }
    // Skip lessons that have only companion files (no main body).
    const grouped = new Map();
    for (const entry of lessonMap.values()) {
      if (!entry.sources.some((s) => s.role === "main")) continue;
      if (!grouped.has(entry.group)) grouped.set(entry.group, []);
      grouped.get(entry.group).push(entry);
    }
    // Natural sort: split into numeric and non-numeric chunks so "100-foo"
    // sorts after "20-foo", not after "10-foo". Keeps outline order intact
    // for books with more than 99 chapters and LeetCode problem numbers.
    const naturalCompare = (a, b) => {
      const parts = (s) => s.split(/(\d+)/).filter(Boolean);
      const pa = parts(a), pb = parts(b);
      for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
        const sa = pa[i], sb = pb[i];
        if (sa === undefined) return -1;
        if (sb === undefined) return 1;
        const na = Number(sa), nb = Number(sb);
        if (!Number.isNaN(na) && !Number.isNaN(nb)) {
          if (na !== nb) return na - nb;
        } else if (sa !== sb) {
          return sa < sb ? -1 : 1;
        }
      }
      return 0;
    };
    for (const arr of grouped.values()) arr.sort((a, b) => naturalCompare(a.item, b.item));

    const subjectOut = path.join(OUT_ROOT, subject.slug);
    await fs.mkdir(subjectOut, { recursive: true });

    const groupSummaries = {};
    for (const [groupSlug, groupItems] of grouped) {
      const groupCfg = subject.groups.find((g) => g.slug === groupSlug);
      const groupLabel = groupCfg?.name || groupSlug;
      const groupNum = groupCfg?.num || groupSlug.slice(0, 2);
      const groupDir = path.join(subjectOut, groupSlug);
      await fs.mkdir(groupDir, { recursive: true });

      const lessonSummaries = [];
      const renderer = null;
      for (let i = 0; i < groupItems.length; i++) {
        const it = groupItems[i];
        // Sort sources so the main body always comes first.
        const orderedSources = [...it.sources].sort((a, b) => {
          if (a.role === "main" && b.role !== "main") return -1;
          if (b.role === "main" && a.role !== "main") return 1;
          return 0;
        });
        const mainSrc = orderedSources.find((s) => s.role === "main") || orderedSources[0];
        const raw = await fs.readFile(mainSrc.absPath, "utf8");

        let parsed;
        let htmlBody;
        if (mainSrc.sourcePath.endsWith(".java")) {
          const javaParsed = parseJavaLesson(raw);
          const question = await loadLeetCodeQuestion(javaParsed.leetcodeSlug);
          const rendered = renderJavaLesson({
            folderTitle: it.item.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
            parsed: javaParsed,
            question,
          });
          parsed = {
            title: rendered.title,
            subtitle: rendered.subtitle,
            meta: rendered.meta,
            body: "",
            toc: rendered.toc,
          };
          htmlBody = rendered.htmlBody;
        } else if (it.isNotebook || mainSrc.sourcePath.endsWith(".ipynb")) {
          try {
            const nb = JSON.parse(raw);
            parsed = renderNotebook(nb, subject, it.imageBase);
            htmlBody = rewriteRawImageSrcs(parsed.prerendered || "", subject, it.imageBase);
          } catch (e) {
            console.warn(`  ! failed to parse notebook ${mainSrc.sourcePath}: ${e.message}`);
            parsed = { title: it.item, subtitle: null, meta: {}, body: "", toc: [] };
            htmlBody = `<pre><code>${escapeHtml(raw)}</code></pre>`;
          }
        } else {
          parsed = parseLesson(raw);
          const localRenderer = makeRenderer(subject, it.imageBase);
          marked.use({ renderer: localRenderer, gfm: true });
          const rawHtml = marked.parse(parsed.body || raw);
          htmlBody = rewriteRawImageSrcs(rawHtml, subject, it.imageBase);
        }

        // Append companion files (e.g. CODE.md) as their own H2 sections.
        for (const src of orderedSources) {
          if (src === mainSrc) continue;
          const heading = src.role === "code" ? "Code walkthrough"
            : src.role.charAt(0).toUpperCase() + src.role.slice(1);
          const compRaw = await fs.readFile(src.absPath, "utf8");
          const localRenderer = makeRenderer(subject, it.imageBase);
          marked.use({ renderer: localRenderer, gfm: true });
          // Strip any leading H1 in the companion so it doesn't fight the main title.
          const stripped = compRaw.replace(/^\s*#\s+.*$/m, "").trimStart();
          const compHtml = marked.parse(stripped);
          const rewrittenComp = rewriteRawImageSrcs(compHtml, subject, it.imageBase);
          const headingId = slugify(heading);
          htmlBody += `\n<h2 id="${headingId}"><a class="anchor" href="#${headingId}">#</a> ${escapeHtml(heading)}</h2>\n${rewrittenComp}`;
          parsed.toc.push({ text: heading, id: headingId });
        }
        htmlBody = embedYouTube(htmlBody);

        const itemNum = (it.item.match(/^(\d+)/)?.[1]) || it.item.slice(0, 2);
        const prevL = groupItems[i - 1];
        const nextL = groupItems[i + 1];
        const prev = prevL ? {
          href: `./${prevL.item}.html`,
          title: prevL.item.replace(/^\d+-/, "").replace(/-/g, " "),
        } : null;
        const next = nextL ? {
          href: `./${nextL.item}.html`,
          title: nextL.item.replace(/^\d+-/, "").replace(/-/g, " "),
        } : null;
        const { owner, name, branch } = subject.repo;
        const sourceUrl = `https://github.com/${owner}/${name}/blob/${branch}/${it.sourcePath.split("/").map(encodeURIComponent).join("/")}`;

        const flagshipKey = `${subject.slug}/${groupSlug}/${it.item}`;
        const outFile = path.join(groupDir, `${it.item}.html`);
        const isFlagship = FLAGSHIP.has(flagshipKey);
        let flagshipExists = false;
        if (isFlagship) {
          try { await fs.access(outFile); flagshipExists = true; }
          catch { flagshipExists = false; }
        }
        if (!isFlagship || !flagshipExists) {
          const html = renderPage({
            subject, group: groupSlug, groupLabel,
            item: it.item, itemNum, parsed, htmlBody, prev, next, sourceUrl,
          });
          await fs.writeFile(outFile, html, "utf8");
        }

        lessonSummaries.push({
          slug: it.item,
          num: itemNum,
          title: parsed.title || it.item.replace(/^\d+-/, "").replace(/-/g, " "),
          meta: parsed.meta,
          flagship: isFlagship,
        });
      }

      const groupIdx = renderGroupIndex({ subject, groupLabel, groupSlug, lessons: lessonSummaries });
      await fs.writeFile(path.join(groupDir, "index.html"), groupIdx, "utf8");

      groupSummaries[groupSlug] = { name: groupLabel, num: groupNum, lessons: lessonSummaries };
      total += lessonSummaries.length;
    }

    manifest.subjects[subject.slug] = {
      name: subject.name,
      tagline: subject.tagline,
      layout: subject.layout,
      totalLessons: Object.values(groupSummaries).reduce((n, g) => n + g.lessons.length, 0),
      groups: groupSummaries,
    };
  }

  manifest.totalLessons = total;
  await fs.mkdir(path.dirname(MANIFEST), { recursive: true });
  await fs.writeFile(MANIFEST, JSON.stringify(manifest, null, 2), "utf8");
  console.log(`\nDone. ${total} lessons across ${Object.keys(manifest.subjects).length} subjects.`);
}

main().catch((e) => { console.error(e); process.exit(1); });
