import { getAllMarkdownFiles } from '@/utils/markdown';
import ProjectFilter from "@/components/project-filter"

export default async function ProjectsPage() {
  const projects = await getAllMarkdownFiles('projects');

  return (
    <div className="container mx-auto px-15 py-20">
      <h1 className="text-4xl font-bold mb-8 text-center">Projects</h1>
      <p className="text-center mb-8 text-gray-600 max-w-2xl mx-auto">
        Explore my portfolio of projects showcasing my skills in web development, software engineering, and creative problem-solving. Each project represents a unique challenge and solution.
      </p>
      <ProjectFilter projects={projects} />
    </div>
  );
}
