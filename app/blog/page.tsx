import { getAllMarkdownFiles } from '@/utils/markdown';
import Link from 'next/link';
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export default async function BlogPage() {
  const posts = await getAllMarkdownFiles('blog');

  return (
    <div className="container mx-auto px-4 py-20">
      <h1 className="text-4xl font-bold mb-8">Blog</h1>
      <div className="grid md:grid-cols-2 gap-6">
        {posts.map((post) => (
          <Link key={post.slug} href={`/blog/${post.slug}`} className="group">
            <Card className="h-full transition-colors hover:border-primary">
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
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
