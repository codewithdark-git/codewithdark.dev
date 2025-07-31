import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "../styles/globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import Navigation from "@/components/navigation";
import { SocialLinksDialog } from "@/components/social-links-dialog";
import Link from "next/link";
import { Github, Linkedin } from "lucide-react"; // Ensure you import the Github icon
import localFont from 'next/font/local'

const inter = localFont({
  src: [
    {
      path: '../public/fonts/Inter-Regular.ttf',
      weight: '300',
      style: 'normal',
    },
    {
      path: '../public/fonts/Inter-Medium.ttf',
      weight: '400',
      style: 'normal',
    },
    {
      path: '../public/fonts/Inter-SemiBold.ttf',
      weight: '500',
      style: 'normal',
    },
    {
      path: '../public/fonts/Inter-Bold.ttf',
      weight: '600',
      style: 'normal',
    },
  ],
  variable: "--font-sans",
})

export const metadata: Metadata = {
  title: "Ahsan Umar - AI/ML Engineer & Researcher. ",
  description:
    "Specializing in NLP, Computer Vision, and Large Language Models.",
  icons: {
    icon: "/favicon.png",
    apple: [
      { url: "/apple-icon.png" },
      { url: "/apple-icon-72x72.png", sizes: "72x72" },
      { url: "/apple-icon-114x114.png", sizes: "114x114" },
    ],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <div className="min-h-screen bg-background text-foreground flex flex-col">
            <Navigation />
            <main className="flex-grow">{children}</main>
            <footer className="py-10 border-t">
              <div className="container mx-auto px-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-16 mb-8 justify-items-center">
                  <div className="space-y-4 text-center">
                    <h3 className="text-lg font-semibold">Ahsan Umar</h3>
                    <p className="text-muted-foreground">
                      Software Engineer specializing in Python & Web Development
                    </p>
                  </div>
                  <div className="space-y-4 text-center">
                    <h3 className="text-lg font-semibold">Quick Links</h3>
                    <ul className="space-y-2">
                  <li>
                  <Link
                    href="/projects"
                    className="text-muted-foreground hover:text-primary"
                  >
                    Projects
                  </Link>
                  </li>
                  <li>
                  <Link
                    href="/blog"
                    className="text-muted-foreground hover:text-primary"
                  >
                    Blog
                  </Link>
                  </li>
                  <li>
                  <Link
                    href="/about"
                    className="text-muted-foreground hover:text-primary"
                  >
                    About
                  </Link>
                  </li>
                    </ul>
                  </div>
                  <div className="space-y-6 text-center">
                    <h3 className="text-lg font-semibold">Connect With Me</h3>
                    <div className="flex flex-col space-y-4 items-center">
                      <a
                        href="https://github.com/codewithdark-git"
                        className="flex items-center space-x-3 text-muted-foreground hover:text-primary transition-colors duration-200 group"
                      >
                        <Github className="w-5 h-5 group-hover:scale-110 transition-transform duration-200" />
                        <span>Follow on GitHub</span>
                      </a>
                      <a
                        href="https://linkedin.com/in/codewithdark"
                        className="flex items-center space-x-3 text-muted-foreground hover:text-primary transition-colors duration-200 group"
                      >
                        <Linkedin className="w-5 h-5 group-hover:scale-110 transition-transform duration-200" />
                        <span>Connect on LinkedIn</span>
                      </a>
                      <div className="pt-2">
                        <SocialLinksDialog />
                      </div>
                    </div>
                  </div>
                </div>
                <div className="text-center mt-8 pt-8 border-t border-gray-700">
                  <p className="text-sm text-muted-foreground">
                    © {new Date().getFullYear()} Ahsan Umar. All rights reserved.
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">
                    Built with Next.js 13, TypeScript, Tailwind CSS, and Shadcn UI
                  </p>
                </div>
              </div>
            </footer>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
