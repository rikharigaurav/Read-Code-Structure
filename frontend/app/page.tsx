"use client"
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Github, Search, History, Star } from "lucide-react";
import Link from "next/link";
import axios from "axios";
import { useState } from "react";
import { useRouter } from "next/navigation";
export default function Home() {
    const router = useRouter( )
    const [searchQuery, setSearchQuery] = useState('')
    const [isLoading, setIsLoading] = useState(false)

    const handleSearch = async () => {
      if (!searchQuery.trim()) return

      setIsLoading(true)
      try {
        // GitHub search API requires a GET request, not POST
        // The endpoint should include the query parameter
        const response = await axios
          .post('http://127.0.0.1:8000/github/', {
            repo_url: searchQuery,
          })

        router.push(
          `/explorer?localFilePath=${encodeURIComponent(
            response.data.localPath
          )}&repo=${encodeURIComponent(searchQuery)}`
        )
      } catch (error) {
        console.error('Error searching repositories:', error)
      } finally {
        setIsLoading(false)
      }
    }

  return (
    <div className='min-h-screen bg-background text-foreground'>
      <main className='container mx-auto px-4 py-8'>
        <div className='flex flex-col items-center justify-center space-y-8'>
          <div className='text-center space-y-4'>
            <Github className='w-16 h-16 mx-auto' />
            <h1 className='text-4xl font-bold'>GitHub Repository Explorer</h1>
            <p className='text-muted-foreground'>
              Explore repositories with enhanced development features
            </p>
          </div>

          <Card className='w-full max-w-2xl p-6'>
            <div className='flex gap-2'>
              <Input
                placeholder='Search repositories...'
                className='flex-1'
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              />
              <Button onClick={handleSearch} disabled={isLoading}>
                <Search className='w-4 h-4 mr-2' />
                {isLoading ? 'Searching...' : 'Search'}
              </Button>
            </div>
          </Card>
        </div>
      </main>
    </div>
  )
}