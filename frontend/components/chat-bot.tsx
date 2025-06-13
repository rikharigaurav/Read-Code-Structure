'use client'

import { useState } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  Send,
  Code,
  BookOpen,
  FileSymlink,
  AlertCircle,
  History,
  Trash2,
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { useChatSessions, Message } from '@/hooks/use-chat-session'

export function ChatBot() {
  const [input, setInput] = useState('')
  const [showHistory, setShowHistory] = useState(false)

  const {
    messages,
    chatSessions,
    currentSessionId,
    isLoading,
    error,
    createNewSession,
    loadSession,
    deleteSession,
    handleSend,
  } = useChatSessions()

  const handleSendMessage = async () => {
    if (input.trim() && !isLoading) {
      await handleSend(input)
      setInput('')
    }
  }

  const handleSessionDelete = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    deleteSession(sessionId)
  }

  const handleSessionLoad = (sessionId: string) => {
    loadSession(sessionId)
    setShowHistory(false)
  }

  return (
    <div className='h-full flex flex-col'>
      <div className='flex justify-between items-center mb-4'>
        <h2 className='text-xl font-bold'>Code Assistant</h2>
        <div className='flex gap-2'>
          <Dialog open={showHistory} onOpenChange={setShowHistory}>
            <DialogTrigger asChild>
              <Button variant='outline' size='sm'>
                <History className='w-4 h-4 mr-2' />
                Chat History
              </Button>
            </DialogTrigger>
            <DialogContent className='max-w-md'>
              <DialogHeader>
                <DialogTitle>Chat History</DialogTitle>
              </DialogHeader>
              <div className='max-h-[500px] overflow-y-auto'>
                {chatSessions.length > 0 ? (
                  <div className='space-y-2'>
                    {chatSessions.map((session) => (
                      <Card
                        key={session.id}
                        className={`cursor-pointer hover:bg-accent transition-colors ${
                          session.id === currentSessionId
                            ? 'border-primary'
                            : ''
                        }`}
                        onClick={() => handleSessionLoad(session.id)}
                      >
                        <CardContent className='p-3 flex justify-between items-center'>
                          <div>
                            <p className='font-medium'>
                              {session.messages.length > 1
                                ? session.messages[1].content.substring(0, 50) +
                                  (session.messages[1].content.length > 50
                                    ? '...'
                                    : '')
                                : 'New conversation'}
                            </p>
                            <p className='text-xs text-muted-foreground'>
                              {session.timestamp.toLocaleString()}
                            </p>
                          </div>
                          <Button
                            variant='ghost'
                            size='sm'
                            onClick={(e) => handleSessionDelete(session.id, e)}
                          >
                            <Trash2 className='w-4 h-4 text-destructive' />
                          </Button>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <p className='text-center py-6 text-muted-foreground'>
                    No chat history found
                  </p>
                )}
              </div>
              <div className='flex justify-between mt-2'>
                <Button variant='outline' onClick={() => setShowHistory(false)}>
                  Close
                </Button>
                <Button onClick={createNewSession}>New Chat</Button>
              </div>
            </DialogContent>
          </Dialog>
          <Button variant='outline' size='sm' onClick={createNewSession}>
            New Chat
          </Button>
        </div>
      </div>

      <Card className='flex-1 mb-4 border shadow-md w-full h-full'>
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
                              <p className='whitespace-pre-line'>
                                {message.fields.summary}
                              </p>
                            </CardContent>
                          </Card>
                        </TabsContent>

                        {message.fields.procedural_knowledge && (
                          <TabsContent value='procedural' className='mt-2'>
                            <Card className='border-0 shadow-none bg-background/10'>
                              <CardContent className='p-3'>
                                <div className='text-sm'>
                                  {Array.isArray(
                                    message.fields.procedural_knowledge
                                  ) ? (
                                    <ul className='space-y-2 list-disc pl-5'>
                                      {message.fields.procedural_knowledge.map(
                                        (step, index) => (
                                          <li
                                            key={index}
                                            className='whitespace-pre-line'
                                          >
                                            {step}
                                          </li>
                                        )
                                      )}
                                    </ul>
                                  ) : (
                                    <div className='whitespace-pre-line'>
                                      {message.fields.procedural_knowledge}
                                    </div>
                                  )}
                                </div>
                              </CardContent>
                            </Card>
                          </TabsContent>
                        )}

                        {message.fields.code_solution && (
                          <TabsContent value='code' className='mt-2'>
                            <Card className='border-0 shadow-none bg-background/10'>
                              <CardHeader className='p-3 pb-0'>
                                <div className='flex justify-between items-center'>
                                  <CardTitle className='text-sm'>
                                    Solution Code
                                  </CardTitle>
                                  <Button
                                    variant='ghost'
                                    size='sm'
                                    onClick={() =>
                                      navigator.clipboard.writeText(
                                        message.fields?.code_solution || ''
                                      )
                                    }
                                    className='h-8 px-2 text-xs'
                                  >
                                    Copy
                                  </Button>
                                </div>
                              </CardHeader>
                              <CardContent className='p-3'>
                                <pre className='font-mono text-sm bg-background/20 p-2 rounded-md overflow-x-auto'>
                                  <code>{message.fields.code_solution}</code>
                                </pre>
                              </CardContent>
                            </Card>
                          </TabsContent>
                        )}

                        {message.fields.visualization_query && (
                          <TabsContent value='visualization' className='mt-2'>
                            <Card className='border-0 shadow-none bg-background/10'>
                              <CardHeader className='p-3 pb-0'>
                                <div className='flex justify-between items-center'>
                                  <CardTitle className='text-sm'>
                                    Data Query
                                  </CardTitle>
                                  <Button
                                    variant='ghost'
                                    size='sm'
                                    onClick={() =>
                                      navigator.clipboard.writeText(
                                        message.fields?.visualization_query ||
                                          ''
                                      )
                                    }
                                    className='h-8 px-2 text-xs'
                                  >
                                    Copy
                                  </Button>
                                </div>
                              </CardHeader>
                              <CardContent className='p-3'>
                                <pre className='font-mono text-sm bg-background/20 p-2 rounded-md overflow-x-auto'>
                                  <code>
                                    {message.fields.visualization_query}
                                  </code>
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
            e.key === 'Enter' &&
            !e.shiftKey &&
            !isLoading &&
            handleSendMessage()
          }
          disabled={isLoading}
          className='shadow-sm'
        />
        <Button
          onClick={handleSendMessage}
          disabled={isLoading}
          className='shadow-sm'
        >
          <Send className='w-4 h-4' />
        </Button>
      </div>
    </div>
  )
}
