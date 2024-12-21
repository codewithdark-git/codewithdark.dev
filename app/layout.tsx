import type { Metadata } from "next"
import { Inter } from 'next/font/google'
import '../styles/globals.css'
import { ThemeProvider } from "@/components/theme-provider"
import Navigation from "@/components/navigation"
import { SocialLinksDialog } from "@/components/social-links-dialog"

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-sans",
})

export const metadata: Metadata = {
  title: "Ahsan Khan - Python Developer",
  description: "Portfolio of Ahsan Khan, a Python developer specializing in web development and data analysis.",
  icons: {
    icon: "/favicon.png",
    // You can also add more icon sizes
    apple: [
      { url: "/apple-icon.png" },
      { url: "/apple-icon-72x72.png", sizes: "72x72" },
      { url: "/apple-icon-114x114.png", sizes: "114x114" },
    ],
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <div className="min-h-screen bg-background text-foreground">
            <Navigation />
            {children}
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
