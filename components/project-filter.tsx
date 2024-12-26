// components/project-filter.tsx
"use client"

import { useState } from 'react'
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"
import Link from "next/link"
import ReactMarkdown from 'react-markdown'
import { Github } from 'lucide-react'

interface Project {
  status: string
  githubUrl: any
  title: string
  slug: string
  image: string
  technologies: string[]
  description: string
  category: string
}

interface ProjectFilterProps {
  projects: Project[]
}

const ProjectFilter: React.FC<ProjectFilterProps> = ({ projects }) => {
  const [filter, setFilter] = useState<string | null>(null)

  const categories = Array.from(new Set(projects?.map(project => project.category) ?? []))

  const filteredProjects = filter
    ? projects.filter(project => project.category === filter)
    : projects

  const [searchQuery, setSearchQuery] = useState("")

  const filteredAndSearchedProjects = filteredProjects.filter(project =>
    project.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    project.description.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <div className="container mx-auto">
      {/* Search Bar */}
      <div className="relative w-full max-w-xl mx-auto mb-8">
        <input
          type="text"
          placeholder="Search projects..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-4 py-2 rounded-lg border border-secondary/20 bg-secondary/5 focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
      </div>

      {/* Filter Section with improved styling */}
      <div className="flex flex-wrap gap-3 mb-8 p-4 bg-secondary/10 backdrop-blur-sm rounded-lg shadow-inner">
        <Badge
          variant={filter === null ? "default" : "outline"}
          className="cursor-pointer hover:scale-105 transition-all"
          onClick={() => setFilter(null)}
        >
          All Projects
        </Badge>
        {categories.map(category => (
          <Badge
            key={category}
            variant={filter === category ? "default" : "outline"}
            className="cursor-pointer hover:scale-105 transition-all"
            onClick={() => setFilter(category)}
          >
            {category}
          </Badge>
        ))}
      </div>

      {/* Projects List with improved spacing and animations */}
      <div className="grid gap-8">
        {filteredAndSearchedProjects.map((project) => (
          <Link key={project.slug} href={`/projects/${project.slug}`}>
            <Card className="group overflow-hidden border border-secondary/20 hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5">
              <CardContent className="p-0">
                <div className="flex flex-col md:flex-row">
                  {/* Image Section */}
                  <div className="relative h-72 md:h-auto md:w-2/5 overflow-hidden">
                    <Image
                      src={project.image ? "/placeholder.svg" : 'image not found'}
                      alt={project.title}
                      fill
                      className="object-cover group-hover:scale-110 transition-transform duration-500"
                    />
                  </div>
                  {/* Content Section */}
                  <div className="flex-1 p-8 space-y-4">
                    <h3 className="text-2xl font-bold tracking-tight text-foreground group-hover:text-primary transition-colors">
                      {project.title}
                    </h3>
                    <div className="prose prose-sm dark:prose-invert line-clamp-3">
                      <ReactMarkdown>{project.description}</ReactMarkdown>
                    </div>
                    <div className="flex items-center gap-4">
                      {project.githubUrl && (
                        <a
                          href={project.githubUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:text-primary transition-colors"
                        >
                          <Github className="w-5 h-5" />
                        </a>
                      )}
                      <Badge variant="secondary" className="bg-emerald-400/10 text-emerald-400 font-medium">
                        {project.status || "Completed"}
                      </Badge>
                    </div>
                    <div className="flex flex-wrap gap-2 pt-4 border-t border-secondary/20">
                      {project.technologies.map((tech) => (
                        <Badge key={tech} variant="secondary" className="bg-blue-400/10 text-blue-400 hover:bg-blue-400/20 transition-colors">
                          {tech}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  )
}

export default ProjectFilter