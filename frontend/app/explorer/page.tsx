'use client'
import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import { ResizablePanelGroup, ResizablePanel } from '@/components/ui/resizable'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { FileTree } from '@/components/file-tree'
import { IssuesList } from '@/components/issues-list'
import { CodeViewer } from '@/components/code-viewer'
import { ChatBot } from '@/components/chat-bot'
import { Neo4jGraph } from '@/components/neo4j-graph-visualizer'
import { MarkdownRenderer } from '@/components/markdown'
import { useSearchParams } from 'next/navigation'
import { Card, CardContent } from '@/components/ui/card'
import { AlertCircle, Loader2 } from 'lucide-react'

interface RepoSummaryResponse {
  folder_path: string
  root_summary: string
  file_count: number
  subfolder_count: number
  folder_name: string
  status: string
}

export default function ExplorerPage() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [localFilePath, setLocalFilePath] = useState<string>('')
  const [repo, setRepo] = useState<string>('')
  const [repoSummary, setRepoSummary] = useState<string>('')
  const [summaryLoading, setSummaryLoading] = useState<boolean>(false)
  const [summaryError, setSummaryError] = useState<string | null>(null)

  const searchParams = useSearchParams()

  useEffect(() => {
    const pathFromUrl = searchParams.get('localFilePath')
    const repoFromUrl = searchParams.get('repo')
    console.log("local file path ")
    console.log(localFilePath)

    if (pathFromUrl) {
      setLocalFilePath(pathFromUrl)
    }
    if (repoFromUrl) {
      setRepo(repoFromUrl)
    }

    const fetchRepoSummary = async () => {
      console.log('Fetching summary for path:', pathFromUrl)
      if (!pathFromUrl) return

      setSummaryLoading(true)
      setSummaryError(null)

      try {
        const response = await axios.post<RepoSummaryResponse>(
          'http://localhost:8000/summarize-folder/',
          {
            localRepoPath: pathFromUrl,
          }
        )

        if (response.data.status === 'success') {
          setRepoSummary(response.data.root_summary)
        } else {
          setSummaryError('Failed to fetch repository summary')
        }
      } catch (error) {
        console.error('Error fetching repository summary:', error)

        let errorMessage = 'An unexpected error occurred'

        if (axios.isAxiosError(error)) {
          if (error.response?.status === 404) {
            errorMessage = 'Repository path not found'
          } else if (error.response?.status === 400) {
            errorMessage = 'Invalid repository path'
          } else if (error.response?.status === 500) {
            errorMessage = 'Server error while processing repository'
          } else {
            errorMessage =
              error.response?.data?.detail ||
              error.response?.data?.message ||
              'Failed to fetch repository summary'
          }
        }

        setSummaryError(errorMessage)
      } finally {
        setSummaryLoading(false)
      }
    }

    if (pathFromUrl) {
      fetchRepoSummary()
    }
  }, [searchParams])

  // Memoize the setSelectedFile callback to prevent unnecessary rerenders
  const handleFileSelect = useCallback((file: string | null) => {
    setSelectedFile(file)
  }, [])

  return (
    <div className='h-screen bg-background'>
      <ResizablePanelGroup direction='horizontal'>
        <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
          <div className='h-screen flex flex-col'>
            <div className='p-4 border-b'>
              <h2 className='text-lg font-semibold'>Repository Structure</h2>
            </div>
            <ScrollArea className='flex-1'>
              <div className='p-4'>
                <FileTree
                  onSelect={handleFileSelect}
                  rootPath={localFilePath}
                />
              </div>
            </ScrollArea>
          </div>
        </ResizablePanel>
        <ResizablePanel defaultSize={75}>
          <Tabs defaultValue='issues' className='h-screen flex flex-col'>
            <div className='px-4 m-3 pt-2 border-b'>
              <TabsList className='flex justify-between'>
                <TabsTrigger value='summary'>Repo Summary</TabsTrigger>
                <TabsTrigger value='issues'>Issues</TabsTrigger>
                <TabsTrigger value='code'>Code Viewer</TabsTrigger>
                <TabsTrigger value='chat'>Chat Bot</TabsTrigger>
                <TabsTrigger value='neo4j'>CodeBase Graph</TabsTrigger>
              </TabsList>
            </div>
            <div className='flex-1 p-4'>
              <TabsContent value='summary' className='h-full'>
                {summaryLoading ? (
                  <Card className='h-full border shadow-md'>
                    <CardContent className='flex items-center justify-center h-full'>
                      <div className='flex flex-col items-center space-y-4'>
                        <Loader2 className='w-8 h-8 animate-spin text-primary' />
                        <p className='text-muted-foreground'>
                          Loading repository summary...
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                ) : summaryError ? (
                  <Card className='h-full border shadow-md'>
                    <CardContent className='flex items-center justify-center h-full'>
                      <div className='flex flex-col items-center space-y-4 text-center'>
                        <AlertCircle className='w-8 h-8 text-destructive' />
                        <div>
                          <p className='text-destructive font-medium'>
                            Error Loading Summary
                          </p>
                          <p className='text-muted-foreground text-sm mt-1'>
                            {summaryError}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ) : (
                  <MarkdownRenderer content={repoSummary} />
                )}
              </TabsContent>
              <TabsContent value='issues' className='h-full'>
                <IssuesList repoURL={repo} />
              </TabsContent>
              <TabsContent value='code' className='h-full overflow-hidden'>
                <CodeViewer file={selectedFile} />
              </TabsContent>
              <TabsContent value='chat' className='h-full'>
                <ChatBot />
              </TabsContent>
              <TabsContent value='neo4j' className='h-full'>
                <Neo4jGraph
                  uri={process.env.NEXT_PUBLIC_NEO4J_URI!}
                  user={process.env.NEXT_PUBLIC_NEO4J_USER!}
                  password={process.env.NEXT_PUBLIC_NEO4J_PASSWORD!}
                />
              </TabsContent>
            </div>
          </Tabs>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}
