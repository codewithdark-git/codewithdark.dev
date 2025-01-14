"use client"

import { Plus } from 'lucide-react'
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

export function SocialLinksDialog() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="icon">
          <Plus className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Connect with me</DialogTitle>
          <DialogDescription>
            Choose a platform to connect or view more of my work.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <Button asChild className="w-full">
            <a href="https://github.com/codewithdark-git" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
          </Button>
          <Button asChild className="w-full">
            <a href="https://linkedin.com/in/codewithdark" target="_blank" rel="noopener noreferrer">
              LinkedIn
            </a>
          </Button>
          <Button asChild className="w-full">
            <a href="mailto:codewithdark90@gmail.com">
              Email
            </a>
          </Button>
          <Button asChild variant="outline" className="w-full">
            <a href="https://linktr.ee/codewithdark" target="_blank" rel="noopener noreferrer">
              Linktree
            </a>
          </Button>
          <Button asChild variant="outline" className="w-full">
            <a href="https://github.com/XCollab" target="_blank" rel="noopener noreferrer">
              Join XCollab
            </a>
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

