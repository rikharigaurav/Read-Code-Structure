"use client"
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Github, Search, History, Star, Loader2 } from "lucide-react";
import Link from "next/link";
import axios from "axios";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
    const router = useRouter()
    const [searchQuery, setSearchQuery] = useState('')
    const [isLoading, setIsLoading] = useState(false)

    const handleSearch = async () => {
      if (!searchQuery.trim()) return

      setIsLoading(true)
      try {
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
        setIsLoading(false) // Reset loading state on error
      }
    }

  return (
    <div className='min-h-screen bg-background text-foreground relative'>
      {/* Loading Overlay */}
      {isLoading && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-background border rounded-lg p-8 max-w-md mx-4 text-center space-y-6">
            <div className="flex justify-center">
              <Loader2 className="w-12 h-12 animate-spin text-primary" />
            </div>
            <div className="space-y-3">
              <h3 className="text-lg font-semibold">Processing Repository</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                Parsing the repository may take a few minutes, depending on its size. 
                Please allow some time for the process to complete. Thank you for your patience!
              </p>
            </div>
          </div>
        </div>
      )}

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
                onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSearch()}
                disabled={isLoading}
              />
              <Button onClick={handleSearch} disabled={isLoading || !searchQuery.trim()}>
                {isLoading ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Search className='w-4 h-4 mr-2' />
                )}
                {isLoading ? 'Searching...' : 'Search'}
              </Button>
            </div>
          </Card>
        </div>
      </main>
    </div>
  )
}