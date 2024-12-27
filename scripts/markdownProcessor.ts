import fs from 'fs/promises';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';

export async function processMarkdownFile(filePath: string) {
  const fileContents = await fs.readFile(filePath, 'utf8');
  const { data, content } = matter(fileContents);
  
  const processedContent = await remark()
    .use(html)
    .process(content);
  const contentHtml = processedContent.toString();

  const type = filePath.includes('projects') ? 'project' : 'blog';

  return {
    type,
    data: {
      ...data,
      slug: filePath.split('/').pop()?.replace('.md', ''),
      content: contentHtml,
      title: data.title || 'Default Title',
      technologies: data.technologies || ['Default Technology'],
    },
  };
}

