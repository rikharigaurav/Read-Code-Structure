'use client'

import { useState, useEffect } from 'react'
import { useChat } from '@/hooks/use-chat'
import { list } from 'postcss'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  fields?: {
    summary?: string
    procedural_knowledge?: string[]
    code_solution?: string
    visualization_query?: string
  }
}

export interface ChatSession {
  id: string
  timestamp: Date
  messages: Message[]
}

const INITIAL_MESSAGE: Message = {
  id: '1',
  role: 'assistant',
  content:
    "Hello! I'm here to help you with your code questions. What would you like to know?",
  timestamp: new Date(),
}

export function useChatSessions() {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE])
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string>('')
  const { sendMessage, isLoading, error } = useChat()

  useEffect(() => {
    const savedSessions = localStorage.getItem('chatSessions')
    if (savedSessions) {
      try {
        const parsedSessions = JSON.parse(savedSessions).map(
          (session: any) => ({
            ...session,
            timestamp: new Date(session.timestamp),
            messages: session.messages.map((msg: any) => ({
              ...msg,
              role: msg.role === 'user' ? 'user' : 'assistant',
              timestamp: new Date(msg.timestamp),
            })),
          })
        )
        setChatSessions(parsedSessions)

        if (parsedSessions.length > 0) {
          const latestSession = parsedSessions[0]
          setCurrentSessionId(latestSession.id)
          setMessages(latestSession.messages)
        } else {
          createNewSession()
        }
      } catch (e) {
        console.error('Error parsing saved chat sessions:', e)
        createNewSession()
      }
    } else {
      createNewSession()
    }
  }, [])

  // Save sessions to localStorage whenever they change
  useEffect(() => {
    if (chatSessions.length > 0) {
      localStorage.setItem('chatSessions', JSON.stringify(chatSessions))
    }
  }, [chatSessions])

  // Update messages in the current session
  useEffect(() => {
    if (currentSessionId && messages.length > 1) {
      setChatSessions((prev) =>
        prev.map((session) =>
          session.id === currentSessionId ? { ...session, messages } : session
        )
      )
    }
  }, [messages, currentSessionId])

  const createNewSession = () => {
    const newSessionId = Date.now().toString()
    const newSession: ChatSession = {
      id: newSessionId,
      timestamp: new Date(),
      messages: [INITIAL_MESSAGE],
    }

    setChatSessions((prev) => [newSession, ...prev])
    setCurrentSessionId(newSessionId)
    setMessages([INITIAL_MESSAGE])

    return newSessionId
  }

  const loadSession = (sessionId: string) => {
    const session = chatSessions.find((s) => s.id === sessionId)
    if (session) {
      setMessages(session.messages)
      setCurrentSessionId(sessionId)
      return true
    }
    return false
  }

  const deleteSession = (sessionId: string) => {
    setChatSessions((prev) => prev.filter((s) => s.id !== sessionId))

    if (sessionId === currentSessionId) {
      if (chatSessions.length > 1) {
        const newCurrentSession = chatSessions.find((s) => s.id !== sessionId)
        if (newCurrentSession) {
          setCurrentSessionId(newCurrentSession.id)
          setMessages(newCurrentSession.messages)
        } else {
          createNewSession()
        }
      } else {
        createNewSession()
      }
    }
  }

  const handleSend = async (input: string) => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])

    try {
      const chatResponse = await sendMessage(input)

      if (chatResponse?.fields) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: chatResponse.fields.summary || "I've analyzed your query.",
          timestamp: new Date(),
          fields: {
            summary: chatResponse.fields.summary || '',
            procedural_knowledge: chatResponse.fields.procedural_knowledge,
            code_solution: chatResponse.fields.code_solution,
            visualization_query: chatResponse.fields.visualization_query,
          },
        }

        setMessages((prev) => [...prev, assistantMessage])
        return assistantMessage
      }
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
      return errorMessage
    }
  }

  return {
    messages,
    chatSessions,
    currentSessionId,
    isLoading,
    error,
    createNewSession,
    loadSession,
    deleteSession,
    handleSend,
  }
}
