import { useState } from 'react';  // Import useState for managing like state
import { getMarkdownContent, getAllMarkdownFiles } from '@/utils/markdown';
import { notFound } from "next/navigation"
import { format } from "date-fns"
import { Badge } from "@/components/ui/badge"
import { CalendarIcon, ClockIcon } from 'lucide-react'

export async function generateStaticParams() {
  const posts = await getAllMarkdownFiles('blog');
  return posts?.map((post) => post ? { slug: post.slug } : null).filter(Boolean);
}

export default async function BlogPost({ params }: { params: { slug: string } }) {
    const post = await getMarkdownContent('blog', params.slug);

    if (!post) {
        notFound();
    }

    // State for tracking likes and whether the user has liked
    const [likes, setLikes] = useState(post.likes || 0);
    const [hasLiked, setHasLiked] = useState(false);

    // Handle like button click
    const handleLike = () => {
        if (!hasLiked) {
            setLikes(likes + 1);
            setHasLiked(true);  // Mark as liked so it doesn't increase again
        }
    };

  return (
    <main className="pt-16 pb-8 bg-background">
      <article className="container mx-auto px-4 max-w-3xl">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold mb-4">{post.title}</h1>
          <div className="flex justify-center items-center space-x-4 text-muted-foreground mb-4">
            {/* Like button */}
            <button 
              onClick={handleLike} 
              className={`flex items-center space-x-2 ${hasLiked ? 'text-blue-500' : 'text-gray-500'}`}
            >
              <span>👏</span>
              <span>{likes} claps</span>
            </button>

            <span className="flex items-center">
              <CalendarIcon className="mr-2 h-4 w-4" />
              {format(new Date(post.date), "MMMM d, yyyy")}
            </span>
            <span className="flex items-center">
              <ClockIcon className="mr-2 h-4 w-4" />
              {post.readTime} read
            </span>
          </div>
          <div className="flex justify-center flex-wrap gap-2">
            {post.categories.map((category: string) => (
              <Badge key={category} variant="outline">{category}</Badge>
            ))}
          </div>
        </header>
        <div 
          className="prose prose-lg prose-invert max-w-none"
          dangerouslySetInnerHTML={{ __html: post.content }}
        />
      </article>
    </main>
  );
}
