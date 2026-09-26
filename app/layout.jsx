import "./globals.css";

export const metadata = {
  title: "AI Engineering from First Principles",
  description:
    "A field-manual walkthrough of AI engineering — from math and ML foundations through transformers, LLMs, agents, and production. 511 lessons, 20 phases.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
