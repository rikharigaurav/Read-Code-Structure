import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { Loader2 } from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { dracula } from 'react-syntax-highlighter/dist/cjs/styles/prism'

interface CodeViewerProps {
  file: string | null
}

export function CodeViewer({ file }: CodeViewerProps) {
  const [content, setContent] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchFileContent = async () => {
      // Reset state when file changes
      setContent('')
      console.log('Fetching content for file:', file)
      setError(null)

      // If no file is selected, return early
      if (!file) return

      try {
        setLoading(true)
        // Fetch file content from backend
        const response = await axios.get(
          `http://localhost:8000/file?path=${encodeURIComponent(file)}`
        )
        // Set the file content
        setContent(response.data)
      } catch (err) {
        console.error('Error fetching file content:', err)
        setError(
          err instanceof Error
            ? `Failed to load file: ${err.message}`
            : 'Failed to load file content'
        )
      } finally {
        setLoading(false)
      }
    }

    fetchFileContent()
  }, [file])

  // If no file is selected
  if (!file) {
    return (
      <div className='flex items-center justify-center h-full text-muted-foreground'>
        Select a file to view its contents
      </div>
    )
  }

  // Loading state
  if (loading) {
    return (
      <div className='flex items-center justify-center h-full'>
        <Loader2 className='w-8 h-8 animate-spin text-muted-foreground' />
      </div>
    )
  }

  // Error state
  if (error) {
    return <div className='p-4 text-red-500 text-sm'>Error: {error}</div>
  }

  // Determine file extension for syntax highlighting
  const getLanguageFromExtension = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase()
    const languageMap: { [key: string]: string } = {
      js: 'javascript',
      jsx: 'jsx',
      ts: 'typescript',
      tsx: 'tsx',
      py: 'python',
      java: 'java',
      cpp: 'cpp',
      c: 'c',
      rb: 'ruby',
      go: 'go',
      rs: 'rust',
      html: 'html',
      css: 'css',
      json: 'json',
      md: 'markdown',
    }
    return languageMap[extension || ''] || 'text'
  }

  return (
    <ScrollArea className='w-screen h-screen'>
      <div className='p-2'>
        <SyntaxHighlighter
          language={getLanguageFromExtension(file)}
          style={dracula}
          customStyle={{
            margin: 0,
            borderRadius: '0.375rem',
            fontSize: '0.875rem',
            background: 'transparent', // Ensure no additional background
          }}
          showLineNumbers
        >
          {content}
        </SyntaxHighlighter>
      </div>
    </ScrollArea>
  )
}
