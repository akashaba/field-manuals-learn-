// Turn a PDF (dropped in content/pdfs/<slug>/book.pdf) into the same
// per-lesson markdown files the rest of the pipeline consumes. Chapters
// are split on the PDF's bookmark outline when it has one, otherwise on
// heading-shaped lines. Embedded images are extracted inline.
//
// Run it against a specific subject slug:
//   node scripts/import-pdf.mjs <subject-slug>
// Or, with no argument, imports every subject registered as local-pdf.

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { SUBJECTS } from "../lib/subjects.js";

// pdfjs's ESM build works in Node once we point its worker at a file:// URL.
const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
const req = createRequire(import.meta.url);
const workerPath = req.resolve("pdfjs-dist/legacy/build/pdf.worker.mjs");
pdfjs.GlobalWorkerOptions.workerSrc = pathToFileURL(workerPath).href;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

// ---------- utils ---------------------------------------------------------
function slugify(s) {
  return String(s).toLowerCase().trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

function pad2(n) { return String(n).padStart(2, "0"); }

// Walk the pdfjs outline into a flat list of {title, pageIndex} entries.
async function flattenOutline(pdf, items) {
  if (!items) return [];
  const out = [];
  for (const it of items) {
    let pageIndex = null;
    try {
      if (it.dest) {
        const dest = typeof it.dest === "string" ? await pdf.getDestination(it.dest) : it.dest;
        if (dest) {
          const ref = dest[0];
          if (ref) pageIndex = await pdf.getPageIndex(ref);
        }
      }
    } catch { /* leave page as null */ }
    if (pageIndex !== null) out.push({ title: it.title.trim(), pageIndex });
    if (it.items?.length) {
      const nested = await flattenOutline(pdf, it.items);
      // include first-level nested but not deeper (avoids over-splitting)
      for (const n of nested) if (!out.find((o) => o.pageIndex === n.pageIndex)) out.push(n);
    }
  }
  out.sort((a, b) => a.pageIndex - b.pageIndex);
  return out;
}

// ---------- font detection ------------------------------------------------
const MONO_FONT_RE = /(mono|courier|consolas|menlo|inconsolata|source\s*code|fira\s*(mono|code)|jetbrains|operator\s*mono|deja\s*vu\s*sans\s*mono|ubuntu\s*mono|liberation\s*mono|hack|iosevka|typewriter|andalemono|lucidaconsole|cascadia)/i;
const MATH_FONT_RE = /^(g_)?(cm[a-z]{2,3}|lm[a-z]+|msam|msbm|euler|mtmi|mtsy|mathmi|mathsy|texgyre)/i;
const MATH_CHAR_RE = /[∀∃∈∉∑∏∫∞≤≥≠≈≡⇒⇔→←↔∧∨¬∪∩⊂⊃⊆⊇∂∇√±×÷∘•·°′″α-ωΑ-Ω⟨⟩⌈⌉⌊⌋]/;

function isMonoFont(name) { return !!name && MONO_FONT_RE.test(name); }
function isMathFont(name) { return !!name && MATH_FONT_RE.test(name); }

// ---------- token → structured rows ---------------------------------------
// Group positioned tokens into rows by y-coordinate, then split each row into
// cells by horizontal gaps larger than the local word-spacing. Returns
// [{ y, cells, tokens, allMono, isMath }] ordered top-to-bottom.
function itemsToRows(items) {
  if (!items.length) return [];
  const yTol = 2;
  const raw = items.filter((it) => it.str && it.str.trim() !== "").map((it) => ({
    x: it.transform?.[4] ?? 0,
    y: it.transform?.[5] ?? 0,
    w: it.width || 0,
    text: it.str,
    font: it.fontFamily || it.fontName || "",
  }));

  const rows = [];
  for (const t of raw) {
    let row = null;
    for (const r of rows) if (Math.abs(r.y - t.y) < yTol) { row = r; break; }
    if (!row) { row = { y: t.y, tokens: [] }; rows.push(row); }
    row.tokens.push(t);
  }
  rows.sort((a, b) => b.y - a.y); // pdfjs y increases upward → top rows first
  for (const r of rows) r.tokens.sort((a, b) => a.x - b.x);

  // Annotate rows with mono/math flags before cell-grouping.
  for (const r of rows) {
    r.allMono = r.tokens.length > 0 && r.tokens.every((t) => isMonoFont(t.font));
    const mathHits = r.tokens.filter((t) => isMathFont(t.font) || MATH_CHAR_RE.test(t.text)).length;
    r.isMath = mathHits > 0 && mathHits >= r.tokens.length * 0.4 && !r.allMono;
  }

  // Merge tokens into cells based on horizontal gap.
  return rows.map((r) => {
    const cells = [];
    if (r.tokens.length === 0) return { y: r.y, cells: [], tokens: [], allMono: false, isMath: false };
    let cur = { x: r.tokens[0].x, endX: r.tokens[0].x + r.tokens[0].w, text: r.tokens[0].text };
    for (let i = 1; i < r.tokens.length; i++) {
      const t = r.tokens[i];
      const gap = t.x - cur.endX;
      if (gap > 8) {
        cells.push(cur);
        cur = { x: t.x, endX: t.x + t.w, text: t.text };
      } else {
        cur.text += (gap > 0.5 ? " " : "") + t.text;
        cur.endX = t.x + t.w;
      }
    }
    cells.push(cur);
    return { y: r.y, cells, tokens: r.tokens, allMono: r.allMono, isMath: r.isMath };
  });
}

// Detect groups of consecutive rows that share a column layout, and turn
// them into Markdown tables. Returns { output: string[], tableSpans: Set<idx> }
function detectTables(rows) {
  const tableSpans = new Set();
  const spans = []; // { start, end } inclusive

  const isCandidate = (row) => {
    if (!row) return false;
    if (row.cells.length < 2 || row.cells.length > 8) return false;
    // reject rows where any cell is very long — probably wrapped prose
    if (row.cells.some((c) => c.text.length > 80)) return false;
    // reject TOC dot-leaders — both "...." runs and ". . . ." spaced variants
    if (row.cells.some((c) => /\.{4,}/.test(c.text) || /(\.\s){3,}/.test(c.text))) return false;
    // reject rows where all cells are ultra-short (diagram element labels)
    if (row.cells.every((c) => c.text.trim().length < 3)) return false;
    // reject rows dominated by non-alphanumeric chars (math notation, braces)
    const total = row.cells.reduce((n, c) => n + c.text.length, 0);
    const alpha = row.cells.reduce((n, c) => n + (c.text.match(/[a-zA-Z0-9]/g) || []).length, 0);
    if (total > 0 && alpha / total < 0.4) return false;
    // require at least one cell of >= 4 chars containing letters
    if (!row.cells.some((c) => c.text.trim().length >= 4 && /[a-zA-Z]/.test(c.text))) return false;
    return true;
  };

  const columnsAlign = (a, b, tolerance = 12) => {
    if (Math.abs(a.cells.length - b.cells.length) > 1) return false;
    const n = Math.min(a.cells.length, b.cells.length);
    for (let i = 0; i < n; i++) {
      if (Math.abs(a.cells[i].x - b.cells[i].x) > tolerance) return false;
    }
    return true;
  };

  let i = 0;
  while (i < rows.length) {
    if (!isCandidate(rows[i])) { i++; continue; }
    let j = i + 1;
    while (j < rows.length && isCandidate(rows[j]) && columnsAlign(rows[i], rows[j])) j++;
    if (j - i >= 3) {
      spans.push({ start: i, end: j - 1 });
      for (let k = i; k < j; k++) tableSpans.add(k);
      i = j;
    } else {
      i++;
    }
  }
  return { spans, tableSpans };
}

function renderTableRows(rows, start, end) {
  const slice = rows.slice(start, end + 1);
  const width = Math.max(...slice.map((r) => r.cells.length));
  const cellsOf = (row) => {
    const out = row.cells.map((c) => c.text.trim());
    while (out.length < width) out.push("");
    return out.map((c) => c.replace(/\|/g, "\\|"));
  };
  const header = cellsOf(slice[0]);
  const sep = header.map(() => "---");
  const body = slice.slice(1).map((r) => "| " + cellsOf(r).join(" | ") + " |");
  return "\n" + [
    "| " + header.join(" | ") + " |",
    "| " + sep.join(" | ") + " |",
    ...body,
  ].join("\n") + "\n";
}

// Render a run of mono-font rows as a fenced code block. Reconstructs
// indentation from x-coordinate: the leftmost row anchors column 0.
function renderCodeRows(rows, start, end) {
  const slice = rows.slice(start, end + 1);
  const leftX = Math.min(...slice.map((r) => r.tokens[0]?.x ?? 0));
  const CHAR_PX = 5.5; // rough width of a mono char at typical PDF font size
  const lines = slice.map((r) => {
    if (!r.tokens.length) return "";
    const indent = Math.max(0, Math.round((r.tokens[0].x - leftX) / CHAR_PX));
    // Preserve inline whitespace by walking tokens and inserting spaces
    // where the horizontal gap exceeds one char-width.
    let text = "";
    let lastEnd = null;
    for (const t of r.tokens) {
      if (lastEnd !== null) {
        const gap = t.x - lastEnd;
        // Adjacent tokens (near-zero gap) get no space. Only insert spaces
        // when the horizontal gap is at least half a char-width.
        const spaces = gap < CHAR_PX * 0.5 ? 0 : Math.round(gap / CHAR_PX);
        text += " ".repeat(spaces);
      }
      text += t.text;
      lastEnd = t.x + t.w;
    }
    return " ".repeat(indent) + text.trimEnd();
  });
  return "\n```\n" + lines.join("\n") + "\n```\n";
}

// Join a word broken by a line-end hyphen (soft-hyphen or ASCII dash) with
// the continuation on the next line. Applied inside the paragraph merger.
function stripLineBreakHyphen(buf, next) {
  if (/[a-z][‐\-]$/.test(buf) && /^[a-z]/.test(next)) {
    return buf.slice(0, -1) + next;
  }
  return buf + " " + next;
}

// Re-flow text: pdfjs gives us positioned tokens; we detect code blocks,
// tables, and math rows, then glue the remaining rows into paragraphs.
function tokensToMarkdown(items) {
  const rows = itemsToRows(items);
  if (!rows.length) return "";

  const { spans, tableSpans } = detectTables(rows);
  const spanByStart = new Map();
  for (const s of spans) spanByStart.set(s.start, s);

  const emitted = [];
  let i = 0;
  while (i < rows.length) {
    // 1) Tables take priority over code detection.
    if (spanByStart.has(i)) {
      const s = spanByStart.get(i);
      emitted.push(renderTableRows(rows, s.start, s.end));
      i = s.end + 1;
      continue;
    }
    // 2) Code blocks — consecutive rows in a monospace font.
    if (rows[i].allMono) {
      let j = i;
      while (j < rows.length && rows[j].allMono && !spanByStart.has(j)) j++;
      // Require ≥ 2 rows to count as a block (single mono line often noise).
      if (j - i >= 2) {
        emitted.push(renderCodeRows(rows, i, j - 1));
        i = j;
        continue;
      }
    }
    // 3) Math rows — preserve verbatim on their own line.
    if (rows[i].isMath) {
      const text = rows[i].cells.map((c) => c.text).join(" ").trim();
      if (text) emitted.push(`\n<div class="pdf-math">${text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>\n`);
      i++;
      continue;
    }
    if (tableSpans.has(i)) { i++; continue; }
    // 4) Prose row.
    const text = rows[i].cells.map((c) => c.text).join(" ").trim();
    if (text) emitted.push(text);
    i++;
  }

  // Merge line-broken paragraphs (skip lines that ARE code, tables, or math).
  const paragraphs = [];
  let buf = "";
  const flush = () => { if (buf) { paragraphs.push(buf); buf = ""; } };
  const isBlock = (s) => s.startsWith("\n|") || s.startsWith("\n```") || s.startsWith("\n<div class=\"pdf-math\"");
  for (const line of emitted) {
    if (isBlock(line)) { flush(); paragraphs.push(line); continue; }
    if (!line) { flush(); continue; }
    const looksHeading = line.length < 90 && /^[A-Z0-9]/.test(line) &&
      (line === line.toUpperCase() || /^(Chapter|Part|Section|Appendix)\b/i.test(line));
    if (looksHeading) { flush(); paragraphs.push(line); continue; }
    if (!buf) buf = line;
    else if (/[.?!:)"]$/.test(buf)) { paragraphs.push(buf); buf = line; }
    else buf = stripLineBreakHyphen(buf, line);
  }
  flush();

  return paragraphs
    .map((p) => {
      if (p.startsWith("\n|") || p.startsWith("\n```") || p.startsWith("\n<div")) return p;
      if (/^(Chapter|Part|Section)\s+\d+/i.test(p) || (p.length < 80 && p === p.toUpperCase() && /[A-Z]/.test(p))) {
        return `## ${p.replace(/\.$/, "")}\n`;
      }
      return p + "\n";
    })
    .join("\n");
}

// ---------- image extraction ---------------------------------------------
// Iterate the operator list for each page, find image XObjects, and write
// them to disk. Wrapped in per-image timeouts because pdfjs's objs.get() can
// hang forever on images that were never materialized (common for JPEG XObjects
// that live in commonObjs, or streams pdfjs skips in Node). Best-effort.
async function extractImages(pdf, imageOutDir, opts = {}) {
  const { perImageTimeoutMs = 2000, pageLimit = null } = opts;
  await fs.mkdir(imageOutDir, { recursive: true });
  const results = [];
  const OPS = pdfjs.OPS;
  let failedTimeouts = 0;

  const pageMax = pageLimit ? Math.min(pageLimit, pdf.numPages) : pdf.numPages;
  for (let pageIndex = 0; pageIndex < pageMax; pageIndex++) {
    const page = await pdf.getPage(pageIndex + 1);
    let ops;
    try { ops = await page.getOperatorList(); }
    catch { page.cleanup(); continue; }
    let imageIndex = 0;
    for (let i = 0; i < ops.fnArray.length; i++) {
      const fn = ops.fnArray[i];
      if (fn !== OPS.paintImageXObject && fn !== OPS.paintJpegXObject) continue;
      const args = ops.argsArray[i];
      const name = args?.[0];
      if (!name) continue;

      const getFromEither = (n) => new Promise((resolve) => {
        try {
          if (page.objs.has && page.objs.has(n))            return resolve(page.objs.get(n));
          if (page.commonObjs.has && page.commonObjs.has(n)) return resolve(page.commonObjs.get(n));
          page.objs.get(n, (o) => resolve(o));
        } catch { resolve(null); }
      });
      const withTimeout = (p, ms) => Promise.race([
        p,
        new Promise((r) => setTimeout(() => r(null), ms)),
      ]);

      const imgObj = await withTimeout(getFromEither(name), perImageTimeoutMs);
      if (!imgObj) { failedTimeouts++; continue; }

      const outName = `page-${pad2(pageIndex + 1)}-${pad2(++imageIndex)}.png`;
      const outPath = path.join(imageOutDir, outName);
      try {
        const png = await imageToPng(imgObj);
        if (png && png.length > 0) {
          await fs.writeFile(outPath, png);
          results.push({ pageIndex, imageIndex, filename: outName });
        }
      } catch { /* skip unsupported */ }
    }
    page.cleanup();
  }
  if (failedTimeouts) console.log(`[pdf]   (${failedTimeouts} images skipped due to timeout — normal for JPEG-heavy PDFs)`);
  return results;
}

// Convert a pdfjs image object to a PNG buffer using zlib deflate.
// Supports grayscale, RGB, and RGBA. JPEG images pass through.
async function imageToPng(img) {
  if (img.data instanceof Uint8Array && img.data.length >= 3 && img.data[0] === 0xff && img.data[1] === 0xd8) {
    // already a JPEG file — write raw
    return Buffer.from(img.data);
  }
  const { width, height, kind, data } = img;
  if (!width || !height || !data) return null;
  let channels;
  switch (kind) {
    case 1: channels = 1; break; // GRAYSCALE_1BPP or GRAYSCALE_8BPP — pdfjs sometimes labels
    case 2: channels = 3; break; // RGB_24BPP
    case 3: channels = 4; break; // RGBA_32BPP
    default: {
      // guess based on data length
      const bpp = data.length / (width * height);
      channels = bpp >= 4 ? 4 : bpp >= 3 ? 3 : 1;
    }
  }
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const src = (y * width + x) * channels;
      const dst = (y * width + x) * 4;
      if (channels === 1) {
        rgba[dst] = rgba[dst + 1] = rgba[dst + 2] = data[src];
        rgba[dst + 3] = 255;
      } else if (channels === 3) {
        rgba[dst] = data[src]; rgba[dst + 1] = data[src + 1]; rgba[dst + 2] = data[src + 2]; rgba[dst + 3] = 255;
      } else {
        rgba[dst] = data[src]; rgba[dst + 1] = data[src + 1]; rgba[dst + 2] = data[src + 2]; rgba[dst + 3] = data[src + 3];
      }
    }
  }
  return encodePng(width, height, rgba);
}

// Minimal PNG encoder — RGBA only, one huge IDAT.
async function encodePng(width, height, rgba) {
  const zlib = await import("node:zlib");
  const crc32 = (buf) => {
    let c = ~0;
    for (let i = 0; i < buf.length; i++) {
      c ^= buf[i];
      for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
    }
    return ~c >>> 0;
  };
  const raw = Buffer.alloc((width * 4 + 1) * height);
  let off = 0;
  for (let y = 0; y < height; y++) {
    raw[off++] = 0; // filter: none
    raw.set(rgba.subarray(y * width * 4, (y + 1) * width * 4), off);
    off += width * 4;
  }
  const idat = zlib.deflateSync(raw, { level: 6 });
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
    const typeBuf = Buffer.from(type, "ascii");
    const crcInput = Buffer.concat([typeBuf, data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(crcInput), 0);
    return Buffer.concat([len, typeBuf, data, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", idat),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

// ---------- import per subject -------------------------------------------
async function importSubject(subject) {
  const bookDir = path.join(ROOT, "content", "pdfs", subject.slug);
  let files;
  try { files = await fs.readdir(bookDir); }
  catch {
    console.warn(`[pdf:${subject.slug}] no folder ${bookDir} — skipping (drop your PDF as ${bookDir}/book.pdf)`);
    return;
  }
  const pdfName = files.find((f) => f.toLowerCase().endsWith(".pdf"));
  if (!pdfName) {
    console.warn(`[pdf:${subject.slug}] no .pdf in ${bookDir} — skipping`);
    return;
  }
  const pdfPath = path.join(bookDir, pdfName);
  console.log(`[pdf:${subject.slug}] loading ${pdfName} ...`);

  const data = new Uint8Array(await fs.readFile(pdfPath));
  const pdf = await pdfjs.getDocument({ data, verbosity: 0 }).promise;
  console.log(`[pdf:${subject.slug}] ${pdf.numPages} pages`);

  const outline = await pdf.getOutline();
  const flat = await flattenOutline(pdf, outline);

  // build chapter spans [{title, start, end}]
  const chapters = [];
  if (flat.length) {
    for (let i = 0; i < flat.length; i++) {
      const start = flat[i].pageIndex;
      const end = i + 1 < flat.length ? flat[i + 1].pageIndex - 1 : pdf.numPages - 1;
      chapters.push({ title: flat[i].title, start, end });
    }
  } else {
    // fallback: whole book is one chapter
    chapters.push({ title: subject.name, start: 0, end: pdf.numPages - 1 });
  }
  console.log(`[pdf:${subject.slug}] ${chapters.length} chapters detected`);

  // extract images once, then map them into chapters by page
  const imageOutDir = path.join(ROOT, "public", "lessons", subject.slug, "chapters", "images");
  const relImageBase = "images/";
  console.log(`[pdf:${subject.slug}] extracting images ...`);
  const images = await extractImages(pdf, imageOutDir);
  console.log(`[pdf:${subject.slug}] extracted ${images.length} images`);

  const rawDir = path.join(ROOT, "content", "raw", subject.slug, "chapters");
  await fs.mkdir(rawDir, { recursive: true });

  for (let i = 0; i < chapters.length; i++) {
    const ch = chapters[i];
    const num = pad2(i + 1);
    const slug = `${num}-${slugify(ch.title) || "chapter"}`;
    const fileName = `${slug}.md`;
    console.log(`[pdf:${subject.slug}] chapter ${num}: "${ch.title}"  pages ${ch.start + 1}-${ch.end + 1}`);

    const pageMds = [];
    for (let p = ch.start; p <= ch.end; p++) {
      const page = await pdf.getPage(p + 1);
      const content = await page.getTextContent();
      // Enrich each item with fontFamily from the page's style dictionary so
      // downstream code can detect monospace (code) and math fonts.
      const styles = content.styles || {};
      const items = content.items.map((it) => ({
        ...it,
        fontFamily: styles[it.fontName]?.fontFamily || it.fontName || "",
      }));
      const md = tokensToMarkdown(items);
      pageMds.push(md);
      // append images that live on this page
      const pageImages = images.filter((im) => im.pageIndex === p);
      for (const im of pageImages) {
        pageMds.push(`\n![](${relImageBase}${im.filename})\n`);
      }
      page.cleanup();
    }

    const heading = `# ${ch.title}\n`;
    const body = pageMds.join("\n\n");
    await fs.writeFile(path.join(rawDir, fileName), heading + "\n" + body, "utf8");
  }

  console.log(`[pdf:${subject.slug}] done. wrote ${chapters.length} chapters to ${rawDir}`);
}

// ---------- main ----------------------------------------------------------
async function main() {
  const targetSlug = process.argv[2];
  const targets = SUBJECTS.filter((s) => s.source === "local-pdf" && (!targetSlug || s.slug === targetSlug));
  if (targets.length === 0) {
    console.log("No local-pdf subjects registered.");
    console.log("To add one:");
    console.log("  1. Add an entry to lib/subjects.js with `source: \"local-pdf\"`.");
    console.log("  2. Place your PDF at content/pdfs/<slug>/book.pdf");
    console.log("  3. Run: npm run content:pdf");
    return;
  }
  for (const s of targets) await importSubject(s);
}

main().catch((e) => { console.error(e); process.exit(1); });
