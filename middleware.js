import { NextResponse } from "next/server";

// HTTP Basic Auth gate for the whole site.
// Set BASIC_AUTH_USER and BASIC_AUTH_PASS in Vercel project env vars.
// If either is unset, the site is open (useful for local dev).
export function middleware(req) {
  const user = process.env.BASIC_AUTH_USER;
  const pass = process.env.BASIC_AUTH_PASS;
  if (!user || !pass) return NextResponse.next();

  const header = req.headers.get("authorization");
  if (header) {
    const [scheme, encoded] = header.split(" ");
    if (scheme === "Basic" && encoded) {
      try {
        const decoded = atob(encoded);
        const idx = decoded.indexOf(":");
        const u = idx >= 0 ? decoded.slice(0, idx) : decoded;
        const p = idx >= 0 ? decoded.slice(idx + 1) : "";
        if (u === user && p === pass) return NextResponse.next();
      } catch { /* fallthrough to 401 */ }
    }
  }

  return new NextResponse("Authentication required.", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Field Manuals", charset="UTF-8"' },
  });
}

// Match every path except Next's own static assets and the favicon.
export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
