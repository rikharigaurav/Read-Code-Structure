'use client'

import { ScrollArea } from '@/components/ui/scroll-area'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  Search,
  AlertCircle,
  MessageSquare,
  CalendarDays,
  User,
  Hash,
  FileSearch,
} from 'lucide-react'
import { useState, useEffect, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import { useRouter } from 'next/navigation'
import { useChatSessions } from '@/hooks/use-chat-session'

interface Issue {
  id: number
  title: string
  number: number
  state: 'open' | 'closed'
  created_at: string
  user: {
    login: string
  }
  body?: string
}

export function IssuesList({ repoURL }: { repoURL: string }) {
  const [selectedIssue, setSelectedIssue] = useState<Issue | null>(null)
  const [issues, setIssues] = useState<Issue[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Get chat session management functions from our custom hook
  const {
    createNewSession,
    handleSend,
    isLoading: isChatLoading,
  } = useChatSessions()

  // Router for navigation
  const router = useRouter()

  const filteredIssues = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    if (!query) return issues

    return issues.filter((issue) => {
      const numberMatch = issue.number.toString() === query
      const titleMatch = issue.title.toLowerCase().includes(query)
      const bodyMatch = issue.body?.toLowerCase().includes(query) || false
      return numberMatch || titleMatch || bodyMatch
    })
  }, [issues, searchQuery])

  useEffect(() => {
    const fetchIssues = async () => {
      try {
        setIsLoading(true)
        setError(null)

        const parsedUrl = new URL(repoURL)
        const pathParts = parsedUrl.pathname
          .split('/')
          .filter((part) => part !== '')

        if (pathParts.length < 2) {
          throw new Error('Invalid GitHub repository URL')
        }

        const owner = pathParts[0]
        const repo = pathParts[1].replace(/\.git$/, '')

        const apiUrl = `https://api.github.com/repos/${owner}/${repo}/issues?state=open`
        const response = await fetch(apiUrl)

        if (!response.ok) {
          throw new Error(`GitHub API error: ${response.status}`)
        }

        const data: Issue[] = await response.json()
        setIssues(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch issues')
      } finally {
        setIsLoading(false)
      }
    }

    if (repoURL) {
      fetchIssues()
    }
  }, [repoURL])

  const handleAskChatbot = async () => {
    if (!selectedIssue || isChatLoading) return

    try {
      const newSessionId = createNewSession()
      const issueMessage = `
        # ${selectedIssue.title} (#${selectedIssue.number})

        ${selectedIssue.body || 'No description provided'}

        ---
        Reported by: ${selectedIssue.user.login}
        State: ${selectedIssue.state}
        Created: ${new Date(selectedIssue.created_at).toLocaleDateString()}
      `
      await handleSend(issueMessage)
    } catch (error) {
      console.error('Failed to process issue for chatbot:', error)
      setError('Failed to send issue to chatbot')
    }
  }

  return (
    <div className='h-full flex flex-col'>
      <div className='mb-4'>
        <div className='relative'>
          <Search className='absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4' />
          <Input
            placeholder='Search issues...'
            className='pl-9'
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {error && <div className='mb-4 text-red-500'>Error: {error}</div>}

      <div className='flex-1 flex gap-4'>
        <Card className='flex-1'>
          <ScrollArea className='h-[600px]'>
            <div className='p-4 space-y-2'>
              {isLoading ? (
                <div className='text-center text-muted-foreground'>
                  Loading issues...
                </div>
              ) : filteredIssues.length > 0 ? (
                filteredIssues.map((issue) => (
                  <div
                    key={issue.id}
                    className={`p-3 rounded-lg cursor-pointer transition-colors ${
                      selectedIssue?.id === issue.id
                        ? 'bg-accent'
                        : 'hover:bg-accent/50'
                    }`}
                    onClick={() => setSelectedIssue(issue)}
                  >
                    <div className='flex items-start gap-2'>
                      <AlertCircle className='w-5 h-5 text-primary mt-0.5' />
                      <div>
                        <h3 className='font-medium'>{issue.title}</h3>
                        <p className='text-sm text-muted-foreground'>
                          #{issue.number} opened by {issue.user.login}
                        </p>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className='text-center text-muted-foreground'>
                  {searchQuery
                    ? 'No matching issues found'
                    : 'No open issues found'}
                </div>
              )}
            </div>
          </ScrollArea>
        </Card>

        <Card className='flex-1'>
          <ScrollArea className='h-[600px]'>
            <div className='p-4'>
              {selectedIssue ? (
                <div>
                  <div className='flex justify-between items-start mb-4 pb-4 border-b border-accent'>
                    <div className='space-y-2'>
                      <h2 className='text-2xl font-bold bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent'>
                        {selectedIssue.title}
                      </h2>
                      <div className='flex items-center gap-2'>
                        <span
                          className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            selectedIssue.state === 'open'
                              ? 'bg-green-100 text-green-800 dark:bg-green-800/30 dark:text-green-300'
                              : 'bg-red-100 text-red-800 dark:bg-red-800/30 dark:text-red-300'
                          }`}
                        >
                          {selectedIssue.state.charAt(0).toUpperCase() +
                            selectedIssue.state.slice(1)}
                        </span>
                        <span className='text-sm text-muted-foreground'>
                          <CalendarDays className='inline mr-1 h-4 w-4' />
                          {new Date(
                            selectedIssue.created_at
                          ).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric',
                          })}
                        </span>
                      </div>
                    </div>
                    <Button
                      variant='secondary'
                      size='sm'
                      onClick={handleAskChatbot}
                      disabled={isChatLoading}
                      className='hover:bg-primary/10 transition-colors'
                    >
                      <MessageSquare className='w-4 h-4 mr-2 text-primary' />
                      <span className='bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent font-semibold'>
                        Ask Chatbot
                      </span>
                    </Button>
                  </div>

                  <div className='space-y-6'>
                    <div className='flex items-center gap-4 text-sm'>
                      <div className='flex items-center gap-1 text-muted-foreground'>
                        <User className='h-4 w-4' />
                        <span className='font-medium text-foreground'>
                          {selectedIssue.user.login}
                        </span>
                      </div>
                      <div className='flex items-center gap-1 text-muted-foreground'>
                        <Hash className='h-4 w-4' />
                        <span className='font-medium text-foreground'>
                          #{selectedIssue.number}
                        </span>
                      </div>
                    </div>

                    <div className='prose dark:prose-invert prose-headings:text-foreground prose-p:text-muted-foreground prose-strong:text-foreground prose-code:bg-accent prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded max-w-none'>
                      {selectedIssue.body ? (
                        <ReactMarkdown>{selectedIssue.body}</ReactMarkdown>
                      ) : (
                        <div className='text-muted-foreground italic'>
                          No description provided
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className='h-full flex items-center justify-center'>
                  <div className='text-center space-y-2'>
                    <FileSearch className='h-8 w-8 mx-auto text-muted-foreground' />
                    <p className='text-muted-foreground font-medium'>
                      Select an issue to view details
                    </p>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </Card>
      </div>
    </div>
  )
}
