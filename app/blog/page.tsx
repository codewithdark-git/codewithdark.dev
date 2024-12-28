import { getAllMarkdownFiles } from '@/utils/markdown';
import Link from 'next/link';
import { format } from 'date-fns';
import { ArrowRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';


export default async function BlogPage() {
  const posts = await getAllMarkdownFiles('blog');

  return (
    <div className="min-h-screen bg-background from-gray-950 via-gray-900 to-black text-white pt-24 pb-16">
      <div className="container mx-auto px-4 max-w-5xl">
      <div className="text-center space-y-4 mb-12">
        <h1 className="text-5xl font-bold bg-gradient-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent">
        Blog Posts
        </h1>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
        Explore my thoughts and insights on web development, programming, and technology
        </p>
      </div>
      <div className="space-y-8">
        {posts.map((post) => (
        <article key={post.slug} className="group hover:bg-gray-800/50 rounded-xl p-6 transition-all duration-300">
          <div className="flex flex-col md:flex-row gap-6 md:gap-8">
          <div className="w-32 flex-shrink-0">
            <time className="block text-sm text-primary font-medium">
            {format(new Date(post.date), "MMM d yyyy")}
            </time>
            <span className="text-sm text-gray-400 font-medium">{post.readTime} read</span>
          </div>
          <div className="flex-1 space-y-4">
            <Link href={`/blog/${post.slug}`} className="block group-hover:transform group-hover:-translate-y-0.5 transition-transform">
            <h2 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent truncate mb-3">
              {post.title}
            </h2>
            <div className="flex flex-wrap gap-2 mb-6">
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
            </Link>
            <Link 
            href={`/blog/${post.slug}`}
            className="inline-flex items-center text-primary hover:text-primary transition-colors font-medium mt-2"
            >
            Read article <ArrowRight className="ml-2 h-4 w-4 animate-pulse" />
            </Link>
          </div>
          </div>
        </article>
        ))}
      </div>
      </div>
    </div>
  );
}

