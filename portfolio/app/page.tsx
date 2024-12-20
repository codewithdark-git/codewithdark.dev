import { Github, Linkedin, Mail, Rocket, Zap, Calendar, MapPin, ArrowRight, Code, Server, Database, PenTool, FileText } from 'lucide-react'
import Image from "next/image"
import Link from "next/link"
import { Button as BaseButton } from "@/components/ui/button"
import { Card as BaseCard, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { SkillsSection } from "@/components/skills-section"
import { getAllMarkdownFiles } from '@/utils/markdown'
import { promises as fs } from 'fs'
import path from 'path'
import { SocialLinksDialog } from "@/components/social-links-dialog"
import { useToast } from "@/hooks/use-toast"

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
  const recentBlogPosts = (await getAllMarkdownFiles('blog')).slice(0, 3)

  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="py-20 md:py-32 bg-gradient-to-br from-background via-muted to-background shadow-premium">
        <div className="container mx-auto px-4 space-y-8">
          <p className="text-primary text-xl font-semibold">Hey there! I&apos;m-</p>
          <h1 className="text-4xl md:text-6xl font-bold text-foreground">Ahsan Khan</h1>
          <div className="space-y-4 max-w-2xl">
            <h2 className="text-2xl md:text-4xl font-semibold text-foreground">
              Python Developer.{" "}
              <span className="text-muted-foreground">
                Specializing in web development and data analysis.
              </span>
            </h2>
            <div className="space-y-2 text-muted-foreground">
              <p className="flex items-center gap-2">
                <Rocket className="text-primary" />
                Currently studying BSAI at Islamia College University Peshawar
              </p>
              <p className="flex items-center gap-2">
                <Mail className="text-primary" />
                codewithdark90@gmail.com
              </p>
              <p className="flex items-center gap-2">
                <Calendar className="text-primary" />
                Born on March 20, 2005
              </p>
              <p className="flex items-center gap-2">
                <MapPin className="text-primary" />
                Peshawar, Pakistan
              </p>
            </div>
            <div className="flex gap-4 pt-4">
              <BaseButton variant="outline" size="icon">
                <Github className="w-4 h-4" />
              </BaseButton>
              <BaseButton variant="outline" size="icon">
                <Linkedin className="w-4 h-4" />
              </BaseButton>
              <BaseButton variant="outline" size="icon">
                <Mail className="w-4 h-4" />
              </BaseButton>
              <SocialLinksDialog />
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="py-20 bg-muted shadow-premium">
        <div className="container mx-auto px-4">
          <div className="flex flex-col lg:flex-row gap-12 items-center">
            <div className="flex-1 space-y-6">
              <h2 className="text-3xl font-bold flex items-center gap-2 text-foreground">
                <Zap className="text-primary" /> About Me
              </h2>
              <p className="text-muted-foreground">
                Ahsan Khan is a Python developer with a passion for creating technology-driven solutions. He specializes in web development and data analysis, with a keen interest in Blockchain and Artificial Intelligence (BSAI).
              </p>
              <p className="text-muted-foreground">
                With experience in various projects, including website development and data analysis, Ahsan is constantly exploring new technologies and methodologies to enhance his skills and deliver innovative solutions.
              </p>
              <p className="text-muted-foreground">
                Currently pursuing a degree in BSAI at Islamia College University Peshawar, Ahsan is excited about the future of technology and his role in shaping it.
              </p>
            </div>
            <div className="lg:flex-1 w-full max-w-md aspect-square relative">
              <div className="absolute inset-0 bg-dot-pattern opacity-20" />
              <Image
                src="/placeholder.svg"
                alt="Ahsan Khan"
                width={400}
                height={400}
                className="rounded-full"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section className="py-20 bg-gradient-to-b from-background to-muted shadow-premium">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold mb-12 text-center text-foreground">My Skills</h2>
          <SkillsSection skillsData={skillsData} />
        </div>
      </section>

      {/* Services Section */}
      <section className="py-20 bg-gradient-to-b from-muted to-background shadow-premium">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold mb-12 text-center text-foreground">Services I Offer</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <BaseCard className="group hover:shadow-lg transition-all duration-300">
              <CardContent className="p-6 text-center">
                <Code className="w-12 h-12 mx-auto mb-4 text-primary" />
                <h3 className="text-xl font-semibold mb-2 text-foreground">Web Development</h3>
                <p className="text-muted-foreground">Custom websites and web applications using modern frameworks and technologies.</p>
              </CardContent>
            </BaseCard>
            <BaseCard className="group hover:shadow-lg transition-all duration-300">
              <CardContent className="p-6 text-center">
                <Server className="w-12 h-12 mx-auto mb-4 text-primary" />
                <h3 className="text-xl font-semibold mb-2 text-foreground">Backend Development</h3>
                <p className="text-muted-foreground">Robust and scalable server-side solutions for your applications.</p>
              </CardContent>
            </BaseCard>
            <BaseCard className="group hover:shadow-lg transition-all duration-300">
              <CardContent className="p-6 text-center">
                <Database className="w-12 h-12 mx-auto mb-4 text-primary" />
                <h3 className="text-xl font-semibold mb-2 text-foreground">Data Analysis</h3>
                <p className="text-muted-foreground">Insightful data analysis and visualization to drive informed decisions.</p>
              </CardContent>
            </BaseCard>
            <BaseCard className="group hover:shadow-lg transition-all duration-300">
              <CardContent className="p-6 text-center">
                <PenTool className="w-12 h-12 mx-auto mb-4 text-primary" />
                <h3 className="text-xl font-semibold mb-2 text-foreground">UI/UX Design</h3>
                <p className="text-muted-foreground">Creating intuitive and visually appealing user interfaces and experiences.</p>
              </CardContent>
            </BaseCard>
          </div>
        </div>
      </section>

      {/* Featured Projects Section */}
      <section className="py-20 bg-gradient-to-b from-background to-muted shadow-premium">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold mb-12 text-center text-foreground">Featured Projects</h2>
          {featuredProjects.length > 0 ? (
            <div className="grid md:grid-cols-2 gap-8 auto-rows-fr">
              {featuredProjects.map((project) => (
                <Link href={`/projects/${project.slug}`} key={project.slug}>
                  <BaseCard className="group hover:shadow-lg transition-all duration-300 transform hover:-translate-y-2">
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
                        <h3 className="text-2xl font-semibold mb-3 group-hover:text-primary transition-colors">{project.title}</h3>
                        <p className="text-muted-foreground mb-4">{project.description}</p>
                        <div className="flex flex-wrap gap-2 mt-auto">
                          {project.technologies.map((tech: string) => (
                            <Badge key={tech} variant="secondary" className="bg-primary/10 text-primary">{tech}</Badge>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </BaseCard>
                </Link>
              ))}
            </div>
          ) : (
            <p className="text-center text-muted-foreground">No projects available at the moment. Check back soon!</p>
          )}
          <div className="mt-12 text-center">
            <BaseButton asChild>
              <Link href="/projects">
                View All Projects <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </BaseButton>
          </div>
        </div>
      </section>

      {/* Recent Blog Posts Section */}
      <section className="py-20 bg-gradient-to-b from-muted to-background shadow-premium">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold mb-12 text-center text-foreground">Recent Blog Posts</h2>
          {recentBlogPosts.length > 0 ? (
            <div className="grid md:grid-cols-3 gap-8">
              {recentBlogPosts.map((post) => (
                <Link href={`/blog/${post.slug}`} key={post.slug}>
                  <BaseCard className="group hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
                    <CardContent className="p-6">
                      <h3 className="text-xl font-semibold mb-2 group-hover:text-primary transition-colors">{post.title}</h3>
                      <div className="text-muted-foreground mb-4">
                        {post.date} • {post.readTime} read
                      </div>
                      <div className="flex gap-2">
                        {post.categories.map((category: string) => (
                          <Badge key={category} variant="secondary">{category}</Badge>
                        ))}
                      </div>
                    </CardContent>
                  </BaseCard>
                </Link>
              ))}
            </div>
          ) : (
            <p className="text-center text-muted-foreground">No blog posts available at the moment. Check back soon!</p>
          )}
          <div className="mt-12 text-center">
            <BaseButton asChild>
              <Link href="/blog">
                View All Blog Posts <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </BaseButton>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="py-20 shadow-premium">
        <div className="container mx-auto px-4 text-center space-y-6">
          <h2 className="text-3xl font-bold text-foreground">Get In Touch</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            I'm always open to new opportunities and collaborations. Feel free to reach out if you'd like to discuss a project or just want to connect!
          </p>
          <div className="flex justify-center gap-4">
            <BaseButton asChild variant="default">
              <a href="mailto:codewithdark90@gmail.com">
                <Mail className="mr-2 h-4 w-4" /> Email Me
              </a>
            </BaseButton>
            <BaseButton asChild variant="outline">
              <a href="/path-to-your-resume.pdf" target="_blank" rel="noopener noreferrer">
                <FileText className="mr-2 h-4 w-4" /> Download Resume
              </a>
            </BaseButton>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t">
        <div className="container mx-auto px-4 text-center text-muted-foreground space-y-4">
          <p>
            © {new Date().getFullYear()} Ahsan Khan. All rights reserved.
          </p>
          <p>
            Built with{" "}
            <Link href="https://nextjs.org" className="text-primary hover:underline">
              Next.js
            </Link>
            {" & "}
            <Link href="https://ui.shadcn.com" className="text-primary hover:underline">
              shadcn/ui
            </Link>
            . Hosted on{" "}
            <Link href="https://vercel.com" className="text-primary hover:underline">
              Vercel
            </Link>
            .
          </p>
        </div>
      </footer>
    </div>
  )
}

