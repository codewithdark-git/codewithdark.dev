import { Github, Linkedin, Mail, Rocket, Zap, Calendar, MapPin, ArrowRight, Code, Server, Database, PenTool, FileText, ChevronRight, ExternalLink } from 'lucide-react'
import Image from "next/image"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { SkillsSection } from "@/components/skills-section"
import { getAllMarkdownFiles } from '@/utils/markdown'
import { promises as fs } from 'fs'
import path from 'path'
import { SocialLinksDialog } from "@/components/social-links-dialog"

export default async function Home() {
  const dataDir = path.join(process.cwd(), 'data')
  const skillsFile = path.join(dataDir, 'skills.json')

  let skillsData: Record<string, { count: number; percentage: number; icon: string }> = {}

  try {
    const fileContent = await fs.readFile(skillsFile, 'utf8')
    skillsData = JSON.parse(fileContent)
  } catch (error) {
    console.error('Error reading skills file:', error)
  }

  const featuredProjects = (await getAllMarkdownFiles('projects')).slice(0, 4)
  const recentBlogPosts = (await getAllMarkdownFiles('blog')).slice(0, 4)

  return (
    <div className="flex flex-col min-h-screen px-4 sm:px-8 md:px-16 lg:px-20 py-10 md:py-20 lg:py-30 bg-background">
      {/* Hero Section */}
      <section className="relative">
      <div className="container mx-auto">
        <div className="max-w-4xl space-y-8 md:space-y-10">
        <div>
          <div className="relative">
          <div className="flex items-center gap-2 mb-4">
          <span className="text-emerald-400 text-2xl">👋</span>
          <span className="text-primary text-3xl font-medium">Hey there! I'm</span>
          </div>
                <h1 className="text-10xl md:text-9xl font-extrabold mt-2 mb-8 relative" lang="en">
                Ahsan Umar.
                {/* Decorative elements */}
                <div className="absolute -top-20 -left-5 w-40 h-40 grid grid-cols-9 gap-2 opacity-20">
                  {[...Array(81)].map((_, i) => (
                  <div 
                  key={i} 
                  className="aspect-square bg-emerald-400/30 rounded-full
                  transform transition-all duration-500 hover:scale-150 hover:bg-emerald-400"
                  />
                  ))}
                </div>
                </h1>
            </div>
          <h2 className="text-3xl font-bold text-gray-300">
                  Software Engineer. <span className="text-gray-500">A self-taught developer with an interest in Computer Science.</span>
                </h2>
        </div>
        
        <div className="space-y-4 text-xl text-muted-foreground">
          <div className="flex items-center gap-2">
            <Rocket className="text-primary" size={30} />
            <span>Currently specializing in Python & Web Development</span>
          </div>
          <div className="flex text-xl items-center gap-2">
            <Zap className="text-primary" size={30} />
            <span>Student at <span className="text-primary" lang="en">Islamia College University</span></span>
          </div>
        </div>
        
        <div className="flex gap-4">
          <Button variant="outline" size="icon" asChild>
            <a href="https://github.com/codewithdark-git" className="text-primary bg-gray-900 hover:bg-gray-800 px-6 py-2 rounded-lg flex items-center gap-2" target="_blank" rel="noopener noreferrer">
              <Github size={20} />
            </a>
          </Button>
          <Button variant="outline" size="icon" asChild>
            <a href="https://linkedin.com/in/codewithdark" className="text-primary bg-gray-900 hover:bg-gray-800 px-6 py-2 rounded-lg flex items-center gap-2" target="_blank" rel="noopener noreferrer">
              <Linkedin size={20} />
            </a>
          </Button>
          <Button variant="outline" size="icon" asChild>
            <a href="mailto:codewithdark90@gmail.com" className="text-primary bg-gray-900 hover:bg-gray-800 px-6 py-2 rounded-lg flex items-center gap-2">
              <Mail size={20} />
            </a>
          </Button>
          <SocialLinksDialog />
        </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="relative py-32">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row items-center gap-16">
        <div className="md:w-1/2 space-y-8">
          <div className="space-y-4">
            <h2 className="text-2xl font-bold flex items-center gap-2 mb-6">
                  <Zap className="text-primary" size={24} />
                  About Me
                </h2>
            <div className="h-1 w-20 bg-primary rounded"></div>
          </div>
          
          <div className="space-y-6">
            <div className="flex items-start gap-4 hover:transform hover:translate-x-2 transition-transform">
          <Code className="w-6 h-6 text-primary mt-1 flex-shrink-0" />
          <p className="text-muted-foreground leading-relaxed">
            A passionate Python developer focused on creating innovative technology solutions, with expertise in web development and data analysis.
          </p>
            </div>

            <div className="flex items-start gap-4 hover:transform hover:translate-x-2 transition-transform">
          <Server className="w-6 h-6 text-primary mt-1 flex-shrink-0" />
          <p className="text-muted-foreground leading-relaxed">
            Experienced in building various projects from website development to complex data analysis systems, always striving to push technological boundaries.
          </p>
            </div>

            <div className="flex items-start gap-4 hover:transform hover:translate-x-2 transition-transform">
          <Database className="w-6 h-6 text-primary mt-1 flex-shrink-0" />
          <p className="text-muted-foreground leading-relaxed">
            Currently pursuing BSAI at Islamia College University Peshawar, focusing on Blockchain and Artificial Intelligence technologies.
          </p>
            </div>
          </div>

          <Button asChild variant="default" className="group hover:text-white">
            <a href="https://drive.google.com/file/d/1hSiqApIaLYrqS3JzCSGrySwuxlbJmjyj/view?usp=sharing" target="_blank" rel="noopener noreferrer">
              <FileText className="mr-2 h-4 w-4 group-hover:scale-110 transition-transform" />
              Download Resume
            </a>
          </Button>
        </div>

        <div className="md:w-1/2 relative group">
          <div className="relative w-[300px] h-[300px] mx-auto">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/30 to-secondary/30 rounded-2xl transform rotate-6 group-hover:rotate-12 transition-transform duration-300"></div>
            <div className="absolute inset-0 bg-gradient-to-tr from-primary/20 to-secondary/20 rounded-2xl transform -rotate-6 group-hover:-rotate-12 transition-transform duration-300"></div>
            <div className="relative bg-background rounded-2xl overflow-hidden shadow-xl">
          <Image
            src="/images/AhsanUmar.png"
            alt="Ahsan Khan"
            width={300}
            height={300}
            className="object-cover transition-transform duration-300 group-hover:scale-105"
          />
            </div>
          </div>
          
            <div className="absolute -bottom-[5%] left-[50%] transform translate-y-1/2 -translate-x-1/2">
              <Badge className="px-6 py-2 text-lg font-medium font-bold text-primary text-black shadow-lg border-2 flex items-center gap-3">
              <Zap className="w-5 h-5 font-bold text-black" />
              Python Developer
              </Badge>
            </div>
        </div>
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section className="relative px-10 py-32 relative overflow-hidden">
        <div className="container mx-auto px-4 relative z-10">
          <div className="text-center space-y-4 mb-16">
        <h2 className="text-4xl font-bold text-foreground flex items-center justify-center gap-2">
          <Code className="text-primary" size={32} />
          Technical Expertise
        </h2>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
          Exploring and mastering a diverse range of technologies, from backend systems to frontend frameworks. 
          Here's my evolving tech stack that powers both professional and personal innovations.
        </p>
          </div>
          <div className="relative">
        <div className="absolute inset-0 via-transparent to-background/50 z-10" />
        <SkillsSection skillsData={skillsData} />
          </div>
        </div>
        <div className="absolute inset-0 bg-grid-white/10 bg-[size:30px_30px] pointer-events-none" />
      </section>

      {/* Featured Projects Section */}
        <section className="relative px-10 py-15 bg-background relative">
          <div className="container mx-auto px-4">
            <div className="text-center space-y-4 mb-16">
          <h2 className="text-4xl font-bold text-foreground flex items-center justify-center gap-2">
            <PenTool className="text-primary" size={32} />
            Featured Projects
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            A collection of my most notable works, showcasing various technologies and problem-solving approaches.
          </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {featuredProjects.slice(0, 4).map((project) => (
            <Link href={`/projects/${project.slug}`} key={project.slug} className="group">
              <Card className="h-full hover:shadow-xl transition-all duration-300 border-2 border-muted overflow-hidden">
            <div className="relative h-48 w-full overflow-hidden">
              <Image
                src={project.image ? "/placeholder.svg" : 'image not found'}
                alt={project.title || "Project Image"}
                fill
                className="object-cover transform group-hover:scale-110 transition-transform duration-300"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent" />
            </div>
            <CardContent className="p-6 space-y-4">
              <div className="flex justify-between items-start">
                <h3 className="text-2xl font-bold group-hover:text-primary transition-colors">
              {project.title.length > 25 ? project.title.slice(0, 25) + "..." : project.title || "Untitled Project"}
                </h3>
                <div className="flex items-center gap-2">
                <div className="flex gap-2">
                  {project.githubUrl && (
                    <Button variant="ghost" size="icon" asChild className="hover:text-primary">
                      <a href={project.githubUrl} target="_blank" rel="noopener noreferrer">
                        <Github size={20} />
                      </a>
                    </Button>
                  )}
                </div>
              <Badge variant="secondary" className="bg-emerald-400/10 text-primary">
                {project.status || "Completed"}
              </Badge>
              <ExternalLink className="text-muted-foreground group-hover:text-primary transition-colors" size={20} />
                </div>
              </div>
              <p className="text-muted-foreground leading-relaxed line-clamp-2">
                {project.description || "No description available"}
              </p>
              <div className="flex flex-wrap gap-2">
                {project.technologies?.map((tech: string) => (
              <Badge 
                key={tech} 
                variant="secondary"
                className="bg-muted-foreground/10 text-muted-foreground hover:bg-emerald-400/10 hover:text-primary transition-colors"
              >
                {tech}
              </Badge>
                ))}
              </div>
            </CardContent>
              </Card>
            </Link>
          ))}
            </div>
            <div className="mt-12 text-center">
          <Button asChild className="group hover:bg-primary hover:text-white">
            <Link href="/projects">
              View All Projects <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-2" />
            </Link>
          </Button>
            </div>
          </div>
        </section>

      {/* Recent Blog Posts Section */}
      <section className="relative px-10 py-20">
        <div className="container mx-auto px-4">
            <div className="text-center space-y-4 mb-16">
            <h2 className="text-4xl font-bold text-foreground flex items-center justify-center gap-2">
              <FileText className="text-primary" size={32} />
              Recent Blog Posts
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Exploring ideas, sharing insights, and documenting my journey in software development.
            </p>
            </div>
          <div className="grid md:grid-cols-2 gap-8">
        {recentBlogPosts.slice(0, 4).map((post) => (
          <Link href={`/blog/${post.slug}`} key={post.slug} className="group">
        <Card className="h-full transition-all duration-300 hover:shadow-xl border-2 border-muted hover:border-primary/50">
          <CardContent className="p-6">
        <h3 className="text-xl font-semibold mb-2 group-hover:text-primary transition-colors">
        {post.title.length > 45 ? post.title.slice(0, 50) + "..." : post.title || "Untitled Post"}
          </h3>
        <p className="text-muted-foreground mb-4 text-sm">
          {post.date} • {post.readTime} read
        </p>
        <div className="flex flex-wrap gap-2 mb-4">
          {post.categories.map((category: string) => (
        <Badge 
          key={category} 
          variant="outline" 
          className="transition-colors duration-300 hover:bg-primary/10 hover:text-primary"
        >
          {category}
        </Badge>
          ))}
        </div>
          </CardContent>
        </Card>
          </Link>
        ))}
          </div>
          <div className="mt-12 text-center">
        <Button asChild className="group hover:bg-primary hover:text-white">
          <Link href="/blog">
        View All Blog Posts <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-2" />
          </Link>
        </Button>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="relative px-10 py-20 bg-background/50">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <div className="text-center space-y-6 mb-12">
              <h2 className="text-4xl font-bold text-foreground flex items-center justify-center gap-2">
                <Mail className="text-primary" size={32} />
                Let's Connect
              </h2>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
                Whether you have a project in mind, a question about my work, or just want to say hello, 
                I'm always excited to connect with fellow developers and potential collaborators.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <Card className="p-6 text-center hover:border-primary/50 transition-colors group">
                <Mail className="mx-auto mb-4 text-primary h-8 w-8 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold mb-2">Email</h3>
                <p className="text-muted-foreground mb-4">Send Your Query</p>
                <Button asChild variant="outline" className="w-full hover:bg-black hover:text-white">
                  <a href="mailto:codewithdark90@gmail.com">
                    Send Email
                  </a>
                </Button>
              </Card>

              <Card className="p-6 text-center hover:border-primary/50 transition-colors group">
                <Linkedin className="mx-auto mb-4 text-primary h-8 w-8 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold mb-2">LinkedIn</h3>
                <p className="text-muted-foreground mb-4">Connect professionally</p>
                <Button asChild variant="outline" className="w-full hover:bg-black hover:text-white">
                  <a href="https://linkedin.com/in/codewithdark" target="_blank" rel="noopener noreferrer">
                    Connect
                  </a>
                </Button>
              </Card>

              <Card className="p-6 text-center hover:border-primary/50 transition-colors group">
                <Github className="mx-auto mb-4 text-primary h-8 w-8 group-hover:scale-110 transition-transform" />
                <h3 className="font-semibold mb-2">GitHub</h3>
                <p className="text-muted-foreground mb-4">Check out my code</p>
                <Button asChild variant="outline" className="w-full hover:bg-black hover:text-white">
                  <a href="https://github.com/codewithdark-git" target="_blank" rel="noopener noreferrer">
                  View Profile
                  </a>
                </Button>
              </Card>
            </div>

            <div className="text-center">
              <Button asChild variant="default" className="group hover:bg-primary hover:text-white">
                <a href="https://drive.google.com/file/d/1hSiqApIaLYrqS3JzCSGrySwuxlbJmjyj/view?usp=sharing" target="_blank" rel="noopener noreferrer">
                  <FileText className="mr-2 h-4 w-4 group-hover:scale-110 transition-transform" />
                  Download Resume
                </a>
              </Button>
            </div>
          </div>
        </div>
        <div className="absolute inset-0 bg-grid-white/10 bg-[size:30px_30px] pointer-events-none" />
      </section>

    </div>
  )
}
