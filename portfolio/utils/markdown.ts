import fs from 'fs/promises';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';

// Define a type for the markdown content
interface MarkdownContent {
  slug: string;
  title: string;
  date: string;
  readTime: string;
  technologies: string[];
  description: string;
}

// Function to get content of a specific markdown file
export async function getMarkdownContent(directory: string, slug: string): Promise<MarkdownContent | null> {
  const fullPath = path.join(process.cwd(), 'content', directory, `${slug}.md`);
  
  try {
    const fileContents = await fs.readFile(fullPath, 'utf8');
    const { data, content } = matter(fileContents);

    const processedContent = await remark()
      .use(html)
      .process(content);
    const contentHtml = processedContent.toString();

    return {
      slug,
      title: data.title || "Untitled", // Default to "Untitled" if title is missing
      date: data.date || "Unknown Date", // Default to "Unknown Date" if date is missing
      readTime: data.readTime || "N/A", // Default to "N/A" if readTime is missing
      technologies: data.technologies || [],
      description: contentHtml,
    };
  } catch (error) {
    console.error(`Error reading markdown file at ${fullPath}:`, error);
    return null; // Return null if there was an error
  }
}

// Function to get all markdown files in a directory
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
          title: data.title || "Untitled", // Ensure title is included
          date: data.date || "Unknown Date", // Ensure date is included
          readTime: data.readTime || "N/A", // Ensure readTime is included
          image: data.image || '/placeholder.svg', // Ensure there's always an image value
          technologies: data.technologies || [],
          categories: data.categories || [], // Ensure categories is an array
        };
      })
    );
  } catch (error) {
    console.error(`Error reading ${directory} directory:`, error);
    return []; // Return an empty array if the directory doesn't exist or is empty
  }
}