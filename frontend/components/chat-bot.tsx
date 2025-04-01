'use client'

import { useState } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Send, Code, BookOpen, FileSymlink, AlertCircle } from 'lucide-react'
import { useChat } from '@/hooks/use-chat'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/badge'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  fields?: {
    summary?: string
    procedural_knowledge?: string
    code_solution?: string
    visualization_query?: string
  }
}

export function ChatBot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content:
        "Hello! I'm here to help you with your code questions. What would you like to know?",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const { sendMessage, response, isLoading, error } = useChat()

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')

    try {
      await sendMessage(input)

      if (response?.fields) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: response.fields.summary || "I've analyzed your query.",
          timestamp: new Date(),
          fields: response.fields,
        }
        setMessages((prev) => [...prev, assistantMessage])
      }
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    }
  }

  return (
    <div className='h-full flex flex-col'>
      <Card className='flex-1 mb-4 border shadow-md'>
        <ScrollArea className='h-[600px]'>
          <div className='p-4 space-y-4'>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[90%] rounded-lg ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground p-3'
                      : 'bg-accent'
                  }`}
                >
                  {message.role === 'assistant' && message.fields ? (
                    <div className='p-3 space-y-3'>
                      <div className='flex items-center space-x-2'>
                        <Badge variant='outline' className='bg-background/20'>
                          AI Response
                        </Badge>
                        <p className='text-xs'>
                          {message.timestamp.toLocaleTimeString()}
                        </p>
                      </div>

                      <div className='font-medium'>
                        {message.fields.summary}
                      </div>

                      <Tabs defaultValue='summary' className='w-full'>
                        <TabsList className='grid grid-cols-4 bg-background/20'>
                          <TabsTrigger value='summary'>Summary</TabsTrigger>
                          <TabsTrigger
                            value='procedural'
                            disabled={!message.fields.procedural_knowledge}
                          >
                            <BookOpen className='w-4 h-4 mr-1' />
                            Steps
                          </TabsTrigger>
                          <TabsTrigger
                            value='code'
                            disabled={!message.fields.code_solution}
                          >
                            <Code className='w-4 h-4 mr-1' />
                            Code
                          </TabsTrigger>
                          <TabsTrigger
                            value='visualization'
                            disabled={!message.fields.visualization_query}
                          >
                            <FileSymlink className='w-4 h-4 mr-1' />
                            Query
                          </TabsTrigger>
                        </TabsList>

                        <TabsContent value='summary' className='mt-2'>
                          <Card className='border-0 shadow-none bg-background/10'>
                            <CardContent className='p-3'>
                              <p>{message.fields.summary}</p>
                            </CardContent>
                          </Card>
                        </TabsContent>

                        {message.fields.procedural_knowledge && (
                          <TabsContent value='procedural' className='mt-2'>
                            <Card className='border-0 shadow-none bg-background/10'>
                              <CardContent className='p-3'>
                                <div className='text-sm whitespace-pre-line'>
                                  {message.fields.procedural_knowledge}
                                </div>
                              </CardContent>
                            </Card>
                          </TabsContent>
                        )}

                        {message.fields.code_solution && (
                          <TabsContent value='code' className='mt-2'>
                            <Card className='border-0 shadow-none bg-background/10'>
                              <CardContent className='p-3'>
                                <pre className='font-mono text-sm bg-background/20 p-2 rounded-md overflow-x-auto'>
                                  {message.fields.code_solution}
                                </pre>
                              </CardContent>
                            </Card>
                          </TabsContent>
                        )}

                        {message.fields.visualization_query && (
                          <TabsContent value='visualization' className='mt-2'>
                            <Card className='border-0 shadow-none bg-background/10'>
                              <CardContent className='p-3'>
                                <pre className='font-mono text-sm bg-background/20 p-2 rounded-md overflow-x-auto'>
                                  {message.fields.visualization_query}
                                </pre>
                              </CardContent>
                            </Card>
                          </TabsContent>
                        )}
                      </Tabs>
                    </div>
                  ) : (
                    <div className='p-3'>
                      <p>{message.content}</p>
                      <p className='text-xs text-muted-foreground mt-1'>
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className='flex justify-start'>
                <div className='bg-accent p-3 rounded-lg'>
                  <div className='flex items-center space-x-2'>
                    <div className='w-2 h-2 bg-primary rounded-full animate-pulse'></div>
                    <div className='w-2 h-2 bg-primary rounded-full animate-pulse delay-75'></div>
                    <div className='w-2 h-2 bg-primary rounded-full animate-pulse delay-150'></div>
                    <span className='text-sm'>Thinking...</span>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className='flex justify-center'>
                <div className='bg-destructive/10 p-3 rounded-lg flex items-center'>
                  <AlertCircle className='w-4 h-4 mr-2 text-destructive' />
                  <span className='text-sm'>{error.message}</span>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </Card>

      <div className='flex gap-2'>
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder='Type your message...'
          onKeyDown={(e) =>
            e.key === 'Enter' && !e.shiftKey && !isLoading && handleSend()
          }
          disabled={isLoading}
          className='shadow-sm'
        />
        <Button onClick={handleSend} disabled={isLoading} className='shadow-sm'>
          <Send className='w-4 h-4' />
        </Button>
      </div>
    </div>
  )
}
