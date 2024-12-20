import { getMarkdownContent, getAllMarkdownFiles } from '@/utils/markdown';
import { notFound } from "next/navigation";
import { format } from "date-fns";

export async function generateStaticParams() {
  const posts = await getAllMarkdownFiles('blog');
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export default async function BlogPost({ params }: { params: { slug: string } }) {
  const post = await getMarkdownContent('blog', params.slug);

  if (!post || !post.title || !post.date) {
    notFound();
  }

  return (
    <main className="pt-16 pb-8">
      <article className="container mx-auto px-4 max-w-3xl">
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-4">{post.title}</h1>
          <div className="text-gray-400">
            {format(new Date(post.date), "MMMM d, yyyy")} • {post.readTime} read
          </div>
        </header>
        <div 
          className="prose prose-invert max-w-none"
          dangerouslySetInnerHTML={{ __html: post.content }}
        />
      </article>
    </main>
  );
}