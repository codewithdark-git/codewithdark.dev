import fs from 'fs/promises';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';

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
    content: contentHtml,
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
          image: data.image || '/placeholder.svg', // Ensure there's always an image value
        };
      })
    );
  } catch (error) {
    console.error(`Error reading ${directory} directory:`, error);
    return []; // Return an empty array if the directory doesn't exist or is empty
  }
}

