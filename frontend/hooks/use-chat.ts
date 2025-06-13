import { useState, useEffect } from 'react'
import axios from 'axios'

interface ChatResponseFields {
  summary: string
  procedural_knowledge?: string[]
  code_solution?: string
  visualization_query?: string
}

export interface ChatResponse {
  fields: ChatResponseFields
}

interface ChatRequest {
  query: string
  context?: Record<string, any>
}

interface UseChatResult {
  sendMessage: (
    query: string,
    context?: ChatRequest['context']
  ) => Promise<ChatResponse>
  response: ChatResponse | null
  isLoading: boolean
  error: Error | null
}

export function useChat(): UseChatResult {
  const [response, setResponse] = useState<ChatResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const sendMessage = async (
    query: string,
    context?: ChatRequest['context']
  ): Promise<ChatResponse> => {
    setIsLoading(true)
    setError(null)

    try {
      const result = await axios.post('http://127.0.0.1:8000/chat/', {
        query: query,
        context: context || {},
      })

      console.log('Chat response:', result.data)

      const chatResponse: ChatResponse = {
        fields: {
          summary: result.data.response.summary || '',
          procedural_knowledge: result.data.response.procedural_knowledge,
          code_solution: result.data.response.code_solution,
          visualization_query: result.data.response.visualization_query,
        },
      }

      setResponse(chatResponse)
      return chatResponse
    } catch (err) {
      const error = err instanceof Error ? err : new Error('An error occurred')
      setError(error)
      console.error('Error sending chat query:', err)
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  return { sendMessage, response, isLoading, error }
}
  