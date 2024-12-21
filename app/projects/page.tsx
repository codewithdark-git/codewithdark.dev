import { getAllMarkdownFiles } from '@/utils/markdown';
import ProjectFilter from "@/components/project-filter"

export default async function ProjectsPage() {
  const projects = await getAllMarkdownFiles('projects');

  return (
    <div className="container mx-auto px-4 py-20">
      <h1 className="text-4xl font-bold mb-8">Projects</h1>
      <ProjectFilter projects={projects} />
    </div>
  );
}
