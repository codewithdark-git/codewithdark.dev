"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { ModeToggle } from "./mode-toggle"

const Navigation = () => {
  const pathname = usePathname()

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center px-4 sm:px-6 lg:px-8">
        <div className="flex-1">
          <Link href="/" className="flex items-center space-x-2">
            <span className="font-bold text-xl">AK</span>
          </Link>
        </div>
        <div className="flex-1 flex justify-center">
          <div className="hidden md:flex items-center space-x-6 text-sm font-medium">
            <Link
              href="/"
              className={cn(
                "transition-colors hover:text-foreground/80",
                pathname === "/" ? "text-foreground" : "text-foreground/60"
              )}
            >
              Home
            </Link>
            <Link
              href="/projects"
              className={cn(
                "transition-colors hover:text-foreground/80",
                pathname?.startsWith("/projects")
                  ? "text-foreground"
                  : "text-foreground/60"
              )}
            >
              Projects
            </Link>
            <Link
              href="/blog"
              className={cn(
                "transition-colors hover:text-foreground/80",
                pathname?.startsWith("/blog")
                  ? "text-foreground"
                  : "text-foreground/60"
              )}
            >
              Blog
            </Link>
            <Link
              href="/contact"
              className={cn(
                "transition-colors hover:text-foreground/80",
                pathname === "/contact"
                  ? "text-foreground"
                  : "text-foreground/60"
              )}
            >
              Contact
            </Link>
          </div>
        </div>
        <div className="flex-1 flex justify-end">
          <ModeToggle />
        </div>
      </div>
    </nav>
  )
}

export default Navigation

