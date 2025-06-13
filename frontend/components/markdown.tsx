'use client'

import React from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Copy, Check } from 'lucide-react'
import { useState } from 'react'

interface MarkdownRendererProps {
  content: string
  className?: string
  showCopyButton?: boolean
}

export function MarkdownRenderer({ 
  content, 
  className = '', 
  showCopyButton = true 
}: MarkdownRendererProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy text: ', err)
    }
  }

  // Simple markdown parser for basic formatting
  const parseMarkdown = (text: string): JSX.Element[] => {
    const lines = text.split('\n')
    const elements: JSX.Element[] = []
    let currentIndex = 0

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]
      
      // Headers
      if (line.startsWith('# ')) {
        elements.push(
          <h1 key={currentIndex++} className="text-2xl font-bold mb-4 text-foreground">
            {line.slice(2)}
          </h1>
        )
      } else if (line.startsWith('## ')) {
        elements.push(
          <h2 key={currentIndex++} className="text-xl font-bold mb-3 text-foreground">
            {line.slice(3)}
          </h2>
        )
      } else if (line.startsWith('### ')) {
        elements.push(
          <h3 key={currentIndex++} className="text-lg font-bold mb-2 text-foreground">
            {line.slice(4)}
          </h3>
        )
      } else if (line.startsWith('#### ')) {
        elements.push(
          <h4 key={currentIndex++} className="text-base font-bold mb-2 text-foreground">
            {line.slice(5)}
          </h4>
        )
      }
      // Code blocks
      else if (line.startsWith('```')) {
        const codeLines: string[] = []
        let j = i + 1
        while (j < lines.length && !lines[j].startsWith('```')) {
          codeLines.push(lines[j])
          j++
        }
        elements.push(
          <Card key={currentIndex++} className="mb-4 border-0 shadow-none bg-accent/50">
            <CardContent className="p-4">
              <pre className="font-mono text-sm bg-background/80 p-3 rounded-md overflow-x-auto border">
                <code className="text-foreground">{codeLines.join('\n')}</code>
              </pre>
            </CardContent>
          </Card>
        )
        i = j // Skip the closing ```
      }
      // Lists
      else if (line.startsWith('- ') || line.startsWith('* ')) {
        const listItems: string[] = [line.slice(2)]
        let j = i + 1
        while (j < lines.length && (lines[j].startsWith('- ') || lines[j].startsWith('* '))) {
          listItems.push(lines[j].slice(2))
          j++
        }
        elements.push(
          <ul key={currentIndex++} className="list-disc pl-6 mb-4 space-y-1">
            {listItems.map((item, idx) => (
              <li key={idx} className="text-foreground">
                {parseInlineMarkdown(item)}
              </li>
            ))}
          </ul>
        )
        i = j - 1
      }
      // Numbered lists
      else if (/^\d+\.\s/.test(line)) {
        const listItems: string[] = [line.replace(/^\d+\.\s/, '')]
        let j = i + 1
        while (j < lines.length && /^\d+\.\s/.test(lines[j])) {
          listItems.push(lines[j].replace(/^\d+\.\s/, ''))
          j++
        }
        elements.push(
          <ol key={currentIndex++} className="list-decimal pl-6 mb-4 space-y-1">
            {listItems.map((item, idx) => (
              <li key={idx} className="text-foreground">
                {parseInlineMarkdown(item)}
              </li>
            ))}
          </ol>
        )
        i = j - 1
      }
      // Blockquotes
      else if (line.startsWith('> ')) {
        const quoteLines: string[] = [line.slice(2)]
        let j = i + 1
        while (j < lines.length && lines[j].startsWith('> ')) {
          quoteLines.push(lines[j].slice(2))
          j++
        }
        elements.push(
          <blockquote key={currentIndex++} className="border-l-4 border-primary pl-4 mb-4 bg-accent/30 py-2 rounded-r-md">
            <p className="text-foreground italic">{quoteLines.join(' ')}</p>
          </blockquote>
        )
        i = j - 1
      }
      // Horizontal rule
      else if (line.trim() === '---' || line.trim() === '***') {
        elements.push(
          <hr key={currentIndex++} className="border-border my-6" />
        )
      }
      // Empty line
      else if (line.trim() === '') {
        elements.push(
          <div key={currentIndex++} className="mb-2"></div>
        )
      }
      // Regular paragraph
      else {
        elements.push(
          <p key={currentIndex++} className="mb-4 text-foreground leading-relaxed">
            {parseInlineMarkdown(line)}
          </p>
        )
      }
    }

    return elements
  }

  // Parse inline markdown (bold, italic, code, links)
  const parseInlineMarkdown = (text: string): React.ReactNode => {
    const parts: React.ReactNode[] = []
    let currentText = text
    let keyCounter = 0

    // Bold text **text**
    currentText = currentText.replace(/\*\*(.*?)\*\*/g, (match, content) => {
      const placeholder = `__BOLD_${keyCounter}__`
      parts.push(<strong key={`bold-${keyCounter++}`} className="font-bold">{content}</strong>)
      return placeholder
    })

    // Italic text *text*
    currentText = currentText.replace(/\*(.*?)\*/g, (match, content) => {
      const placeholder = `__ITALIC_${keyCounter}__`
      parts.push(<em key={`italic-${keyCounter++}`} className="italic">{content}</em>)
      return placeholder
    })

    // Inline code `code`
    currentText = currentText.replace(/`(.*?)`/g, (match, content) => {
      const placeholder = `__CODE_${keyCounter}__`
      parts.push(
        <code key={`code-${keyCounter++}`} className="bg-accent px-1.5 py-0.5 rounded text-sm font-mono border">
          {content}
        </code>
      )
      return placeholder
    })

    // Links [text](url)
    currentText = currentText.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, linkText, url) => {
      const placeholder = `__LINK_${keyCounter}__`
      parts.push(
        <a key={`link-${keyCounter++}`} href={url} className="text-primary hover:underline" target="_blank" rel="noopener noreferrer">
          {linkText}
        </a>
      )
      return placeholder
    })

    // Split by placeholders and reconstruct
    const finalParts: React.ReactNode[] = []
    const textParts = currentText.split(/(__\w+_\d+__)/g)
    
    textParts.forEach((part, index) => {
      if (part.startsWith('__') && part.endsWith('__')) {
        const matchedPart = parts.find((p: any) => 
          p.key && part.includes(p.key.split('-')[0].replace('__', '').toUpperCase())
        )
        if (matchedPart) finalParts.push(matchedPart)
      } else if (part) {
        finalParts.push(part)
      }
    })

    return finalParts.length > 0 ? finalParts : text
  }

  return (
    <Card className={`h-full border shadow-md ${className}`}>
      {showCopyButton && (
        <div className="flex justify-end p-3 pb-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="h-8 px-2 text-xs"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 mr-1" />
                Copied
              </>
            ) : (
              <>
                <Copy className="w-3 h-3 mr-1" />
                Copy
              </>
            )}
          </Button>
        </div>
      )}
      
      <ScrollArea className="h-[600px]">
        <CardContent className="p-6 space-y-2">
          {content ? (
            parseMarkdown(content)
          ) : (
            <p className="text-muted-foreground text-center py-8">
              No content to display
            </p>
          )}
        </CardContent>
      </ScrollArea>
    </Card>
  )
}

export default MarkdownRenderer