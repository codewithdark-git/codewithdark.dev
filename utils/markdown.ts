import fs from 'fs/promises';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';
import { title } from 'process';


export interface Post {
  slug: string;
  title: string;
  date: string;
  readTime: string;
  categories: string[];
  content: string;
  image: string;
}

export interface Project {
  slug: string;
  title: string;
  description: string;
  image: string;
  url: string;
  github: string;
  technologies: string[];
}


export async function getMarkdownContent(directory: string, slug: string) {
  const fullPath = path.join(process.cwd(), 'content', directory, `${slug}.md`);
  const fileContents = await fs.readFile(fullPath, 'utf8');

  const { data, content } = matter(fileContents);

  const processedContent = await remark()
    .use(html)
    .process(content);
  const contentHtml = processedContent.toString();

  return {
    slug,
    ...data,
    date: data.date,
    title: data.title || title,
    description: data.description || '',
    categories: data.categories || [],
    readTime: data.readTime || '',
    content: contentHtml,
    image: data.image || '/placeholder.svg',
    githubUrl: data.github || '', // Ensure there's always a GitHub URL value
    technologies: data.technologies || [],
    liveUrl: data.url || '', // Ensure there's always a URL value

  };
}



export async function getAllMarkdownFiles(directory: string) {
  const fullPath = path.join(process.cwd(), 'content', directory);
  try {
    const files = await fs.readdir(fullPath);

    return Promise.all(
      files.map(async (filename) => {
        const filePath = path.join(fullPath, filename);
        const fileContents = await fs.readFile(filePath, 'utf8');
        const { data } = matter(fileContents);

        return {
          slug: filename.replace('.md', ''),
          ...data,
          date: data.date,
          title: data.title || title,
          technologies: data.technologies || [],
          description: data.description || '',
          liveUrl: data.url || '',
          readTime: data.readTime || '',
          categories: data.categories || [],
          githubUrl: data.github || '',
          image: data.image || '/placeholder.svg', // Ensure there's always an image value
          

        };
      })
    );
  } catch (error) {
    console.error(`Error reading ${directory} directory:`, error);
    return []; // Return an empty array if the directory doesn't exist or is empty
  }
}
