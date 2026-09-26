import fs from "node:fs";
import path from "node:path";
import { notFound } from "next/navigation";
import { SUBJECTS, subjectBySlug } from "@/lib/subjects";

export function generateStaticParams() {
  return SUBJECTS.map((s) => ({ subject: s.slug }));
}

function loadManifest() {
  const p = path.join(process.cwd(), "content", "manifest.json");
  if (!fs.existsSync(p)) return { subjects: {} };
  try { return JSON.parse(fs.readFileSync(p, "utf8")); }
  catch { return { subjects: {} }; }
}

export function generateMetadata({ params }) {
  const subject = subjectBySlug(params.subject);
  if (!subject) return {};
  return { title: `${subject.name} — Field Manual`, description: subject.description };
}

export default function SubjectPage({ params }) {
  const subject = subjectBySlug(params.subject);
  if (!subject) return notFound();
  const manifest = loadManifest();
  const m = manifest.subjects?.[subject.slug] || { groups: {} };

  return (
    <div className="shell" style={{ "--sc": subject.accent, "--sc2": subject.accent2 }}>
      <header className="masthead">
        <div className="brand">
          <a href="/" style={{ color: "var(--ink-soft)" }}>Field Manuals</a>
          &nbsp;›&nbsp;<b>{subject.name}</b>
        </div>
        <nav>
          <a href={`https://github.com/${subject.repo.owner}/${subject.repo.name}`} target="_blank" rel="noreferrer">
            Source
          </a>
        </nav>
      </header>

      <section>
        <div className="eyebrow" style={{ color: "var(--sc)" }}>
          {m.totalLessons ? `${m.totalLessons} lessons` : subject.tocLabel}
        </div>
        <h1 className="title">
          {subject.name},<br />
          <em style={{ color: "var(--sc)" }}>{subject.tagline}.</em>
        </h1>
        <p className="lede">{subject.description}</p>
      </section>

      {subject.layout === "phases" ? (
        <>
          <div id="phases" />
          <div className="section-title">The {subject.groups.length} phases</div>
          <div className="phases">
            {subject.groups.map((g) => {
              const summary = m.groups?.[g.slug] || {};
              const count = summary.lessons?.length || 0;
              const href = count > 0
                ? `/lessons/${subject.slug}/${g.slug}/index.html`
                : `https://github.com/${subject.repo.owner}/${subject.repo.name}/tree/${subject.repo.branch}/phases/${g.slug}`;
              return (
                <a key={g.slug} className={`phase${g.deep ? " deep" : ""}`} href={href} style={{ "--sc": subject.accent }}>
                  <div className="num">PHASE {g.num}</div>
                  <div className="name">{g.name}</div>
                  <div className="meta">
                    {count > 0 && <span>{count} lessons</span>}
                    <span style={{ color: "var(--ink-faint)" }}>{g.blurb}</span>
                  </div>
                  <div className="arrow">→</div>
                </a>
              );
            })}
          </div>
        </>
      ) : (
        <>
          <div className="section-title">Chapters</div>
          <div className="lessons">
            {Object.values(m.groups).flatMap((group) =>
              group.lessons.map((l) => (
                <a
                  key={l.slug}
                  className="lesson"
                  href={`/lessons/${subject.slug}/${Object.keys(m.groups).find(k => m.groups[k] === group)}/${l.slug}.html`}
                >
                  <div className="lnum">{l.num}</div>
                  <div className="lname">{l.title}</div>
                </a>
              ))
            )}
          </div>
        </>
      )}

      <footer className="foot">
        <div>Content by upstream author, MIT-licensed.</div>
        <div>Field-manual presentation by this project.</div>
      </footer>
    </div>
  );
}
