import fs from "node:fs";
import path from "node:path";
import { SUBJECTS } from "@/lib/subjects";

function loadManifest() {
  const p = path.join(process.cwd(), "content", "manifest.json");
  if (!fs.existsSync(p)) return { subjects: {}, totalLessons: 0 };
  try { return JSON.parse(fs.readFileSync(p, "utf8")); }
  catch { return { subjects: {}, totalLessons: 0 }; }
}

export const metadata = {
  title: "Field Manuals",
  description: "Long-form reference series in the field-manual format.",
};

export default function Home() {
  const manifest = loadManifest();
  return (
    <div className="shell">
      <header className="masthead">
        <div className="brand">
          <b>Field Manuals</b> · a shelf of long-form references
        </div>
        <nav>
          <a href="https://github.com/rohitg00/ai-engineering-from-scratch" target="_blank" rel="noreferrer">
            AI source
          </a>
          <a href="https://github.com/liquidslr/system-design-notes" target="_blank" rel="noreferrer">
            System design source
          </a>
        </nav>
      </header>

      <section>
        <div className="eyebrow">A shelf of {manifest.totalLessons || "many"} lessons</div>
        <h1 className="title">
          Learn a subject the way you read a book —<br />
          <em>from first principles.</em>
        </h1>
        <p className="lede">
          Each subject on this shelf is a self-contained field manual: long-form pages,
          hand-crafted diagrams where they matter, live code, and a table of contents you
          can browse end-to-end. Pick one.
        </p>
      </section>

      <div id="subjects" />
      <div className="section-title">Subjects</div>
      <div className="subjects">
        {SUBJECTS.map((s) => {
          const m = manifest.subjects?.[s.slug];
          const lessonCount = m?.totalLessons || 0;
          const groupCount = m ? Object.keys(m.groups).length : (s.groups?.length || 0);
          return (
            <a
              key={s.slug}
              className={`subject-card${s.medium === "book" ? " is-book" : ""}`}
              href={`/${s.slug}`}
              style={{ "--sc": s.accent, "--sc2": s.accent2 }}
            >
              {s.medium === "book" && <div className="sc-badge">BOOK</div>}
              <div className="sc-eyebrow">{s.tocLabel}</div>
              <div className="sc-name">{s.name}</div>
              <div className="sc-tagline">{s.tagline}</div>
              <p className="sc-blurb">{s.description}</p>
              <div className="sc-meta">
                {lessonCount > 0 && <span><b>{lessonCount}</b> {s.medium === "book" ? "chapters" : "lessons"}</span>}
                {groupCount > 0 && s.layout === "phases" && <span><b>{groupCount}</b> phases</span>}
                {s.layout === "flat" && s.medium !== "book" && <span>flat index</span>}
                {s.medium === "book" && <span>PDF import</span>}
              </div>
              <div className="sc-arrow">Open →</div>
            </a>
          );
        })}
      </div>

      <footer className="foot">
        <div>Content from upstream authors, MIT-licensed. Field-manual presentation by this project.</div>
        <div>Add a subject: define it in <code>lib/subjects.js</code>, run <code>npm run content:all</code>.</div>
      </footer>
    </div>
  );
}
