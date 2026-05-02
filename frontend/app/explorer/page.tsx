'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable'
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
import {
  AlertCircle,
  Loader2,
  BookOpen,
  GitBranch,
  Code2,
  MessageSquare,
  NetworkIcon,
  FolderOpen,
  ExternalLink,
} from 'lucide-react'

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
  const [repoName, setRepoName] = useState<string>('')
  const [repoSummary, setRepoSummary] = useState<string>('')
  const [summaryLoading, setSummaryLoading] = useState<boolean>(false)
  const [summaryError, setSummaryError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('issues')

  const summaryFetched = useRef<string | null>(null)
  const searchParams = useSearchParams()

  useEffect(() => {
    const pathFromUrl = searchParams.get('localFilePath')
    const repoFromUrl = searchParams.get('repo')

    if (pathFromUrl) setLocalFilePath(pathFromUrl)
    if (repoFromUrl) {
      setRepo(repoFromUrl)
      // Extract human-readable repo name from URL
      try {
        const parts = new URL(repoFromUrl).pathname.split('/').filter(Boolean)
        setRepoName(parts.length >= 2 ? `${parts[0]}/${parts[1]}` : repoFromUrl)
      } catch {
        setRepoName(repoFromUrl)
      }
    }

    const fetchRepoSummary = async () => {
      if (!pathFromUrl || summaryFetched.current === pathFromUrl) return
      summaryFetched.current = pathFromUrl
      setSummaryLoading(true)
      setSummaryError(null)
      try {
        const response = await axios.post<RepoSummaryResponse>('/api/summarize-folder/', {
          localRepoPath: pathFromUrl,
        })
        if (response.data.status === 'success') {
          setRepoSummary(response.data.root_summary)
        } else {
          setSummaryError('Failed to fetch repository summary')
        }
      } catch (error) {
        summaryFetched.current = null
        let msg = 'An unexpected error occurred'
        if (axios.isAxiosError(error)) {
          if (error.response?.status === 404) msg = 'Repository path not found'
          else if (error.response?.status === 400) msg = 'Invalid repository path'
          else if (error.response?.status === 500) msg = 'Server error while processing repository'
          else msg = error.response?.data?.detail || error.response?.data?.message || 'Failed to fetch repository summary'
        }
        setSummaryError(msg)
      } finally {
        setSummaryLoading(false)
      }
    }

    if (pathFromUrl) fetchRepoSummary()
  }, [searchParams])

  const handleFileSelect = useCallback((file: string | null) => {
    setSelectedFile(file)
    setActiveTab('code')
  }, [])

  const TABS = [
    { value: 'summary', label: 'Summary',  Icon: BookOpen },
    { value: 'issues',  label: 'Issues',   Icon: AlertCircle },
    { value: 'code',    label: 'Code',     Icon: Code2 },
    { value: 'chat',    label: 'Chat',     Icon: MessageSquare },
    { value: 'neo4j',   label: 'Graph',    Icon: NetworkIcon },
  ]

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">

      {/* ── Top header bar ─────────────────────────────────────────────────── */}
      <header className="flex items-center gap-3 px-4 h-11 border-b bg-background/95 backdrop-blur
        supports-[backdrop-filter]:bg-background/60 flex-shrink-0 z-10">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground/90">
          <GitBranch className="w-4 h-4 text-muted-foreground" />
          <span className="text-muted-foreground">CodeExplorer</span>
          {repoName && (
            <>
              <span className="text-muted-foreground/40">/</span>
              <span className="font-semibold text-foreground">{repoName}</span>
              {repo && (
                <a href={repo} target="_blank" rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground transition-colors">
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              )}
            </>
          )}
        </div>
        {selectedFile && (
          <div className="flex items-center gap-1.5 ml-2 text-xs text-muted-foreground">
            <span className="text-muted-foreground/40">·</span>
            <Code2 className="w-3.5 h-3.5" />
            <span className="font-mono truncate max-w-xs">{selectedFile.split(/[/\\]/).pop()}</span>
          </div>
        )}
      </header>

      {/* ── Two-pane body ──────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup direction="horizontal" className="h-full">

          {/* Left: file tree */}
          <ResizablePanel defaultSize={22} minSize={15} maxSize={40}>
            <div className="h-full flex flex-col border-r">
              <div className="flex items-center gap-2 px-4 h-10 border-b bg-muted/20 flex-shrink-0">
                <FolderOpen className="w-4 h-4 text-muted-foreground" />
                <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Files
                </span>
              </div>
              <ScrollArea className="flex-1">
                <div className="p-2">
                  <FileTree onSelect={handleFileSelect} rootPath={localFilePath} />
                </div>
              </ScrollArea>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Right: tabbed content */}
          <ResizablePanel defaultSize={78}>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">

              {/* Tab bar */}
              <div className="flex-shrink-0 border-b bg-background px-2 pt-1">
                <TabsList className="h-9 bg-transparent gap-0 p-0">
                  {TABS.map(({ value, label, Icon }) => (
                    <TabsTrigger
                      key={value}
                      value={value}
                      className="relative h-9 px-4 rounded-none border-b-2 border-transparent text-xs font-medium
                        text-muted-foreground transition-none
                        data-[state=active]:border-primary data-[state=active]:text-foreground
                        data-[state=active]:bg-transparent hover:text-foreground gap-1.5"
                    >
                      <Icon className="w-3.5 h-3.5" />
                      {label}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </div>

              {/* Tab panels */}
              <div className="flex-1 overflow-hidden">

                <TabsContent value="summary" className="h-full m-0 p-4 overflow-auto">
                  {summaryLoading ? (
                    <Card className="h-full border shadow-sm">
                      <CardContent className="flex items-center justify-center h-full">
                        <div className="flex flex-col items-center gap-3">
                          <Loader2 className="w-7 h-7 animate-spin text-primary" />
                          <p className="text-sm text-muted-foreground">Generating repository summary…</p>
                        </div>
                      </CardContent>
                    </Card>
                  ) : summaryError ? (
                    <Card className="h-full border shadow-sm">
                      <CardContent className="flex items-center justify-center h-full">
                        <div className="flex flex-col items-center gap-3 text-center">
                          <AlertCircle className="w-7 h-7 text-destructive" />
                          <div>
                            <p className="text-destructive font-medium text-sm">Error Loading Summary</p>
                            <p className="text-muted-foreground text-xs mt-1">{summaryError}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ) : (
                    <MarkdownRenderer content={repoSummary} />
                  )}
                </TabsContent>

                <TabsContent value="issues" className="h-full m-0 p-4 overflow-hidden">
                  <IssuesList repoURL={repo} />
                </TabsContent>

                <TabsContent value="code" className="h-full m-0 overflow-hidden">
                  <CodeViewer file={selectedFile} />
                </TabsContent>

                <TabsContent value="chat" className="h-full m-0 p-4 overflow-hidden">
                  <ChatBot />
                </TabsContent>

                <TabsContent value="neo4j" className="h-full m-0 overflow-hidden p-2">
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
    </div>
  )
}
