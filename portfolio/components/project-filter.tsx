"use client"

import { useState } from 'react'
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"
import Link from "next/link"

interface Project {
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

  const categories = Array.from(new Set(projects.map(project => project.category)))

  const filteredProjects = filter
    ? projects.filter(project => project.category === filter)
    : projects

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        <Badge
          variant={filter === null ? "default" : "outline"}
          className="cursor-pointer"
          onClick={() => setFilter(null)}
        >
          All
        </Badge>
        {categories.map(category => (
          <Badge
            key={category}
            variant={filter === category ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setFilter(category)}
          >
            {category}
          </Badge>
        ))}
      </div>
      <div className="grid md:grid-cols-2 gap-6">
        {filteredProjects.map((project) => (
          <Link key={project.slug} href={`/projects/${project.slug}`}>
            <Card className="h-full hover:border-primary transition-colors">
              <CardContent className="p-0">
                <div className="relative h-48 w-full">
                  <Image
                    src={project.image || "/placeholder.svg"}
                    alt={project.title}
                    fill
                    className="object-cover rounded-t-lg"
                  />
                </div>
                <div className="p-6">
                  <h3 className="text-xl font-semibold mb-2">{project.title}</h3>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.technologies.map((tech) => (
                      <Badge key={tech} variant="secondary">{tech}</Badge>
                    ))}
                  </div>
                  <p className="text-muted-foreground">{project.description}</p>
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

