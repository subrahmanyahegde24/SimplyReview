import type { Metadata } from "next";
import { Hanken_Grotesk, Newsreader } from "next/font/google";
import "./globals.css";

const ui = Hanken_Grotesk({ variable: "--font-ui", subsets: ["latin"] });
const reading = Newsreader({ variable: "--font-reading", subsets: ["latin"], style: ["normal", "italic"] });

export const metadata: Metadata = {
  title: "SimplyReview",
  description: "Ask questions across your research papers with hybrid search and a knowledge graph.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${ui.variable} ${reading.variable}`}>
      <body>{children}</body>
    </html>
  );
}
