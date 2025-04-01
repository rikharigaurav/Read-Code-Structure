import { useState } from 'react';
import axios from 'axios';
import type { ChatRequest, ChatResponse } from '@/lib/types/api';

interface UseChatResult {
  sendMessage: (
    query: string,
    context?: ChatRequest['context']
  ) => Promise<void>
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
  ) => {
    setIsLoading(true)
    setError(null)

    try {
      const result = await axios.post('http://127.0.0.1:8000/chat/', {
        query: query,
        context: context || {},
      })

      // Parse the response assuming it follows the structure described
      const chatResponse: ChatResponse = {
        fields: {
          summary: result.data.summary || '',
          procedural_knowledge: result.data.procedural_knowledge,
          code_solution: result.data.code_solution,
          visualization_query: result.data.visualization_query,
        },
      }

      setResponse(chatResponse)
    } catch (err) {
      setError(err instanceof Error ? err : new Error('An error occurred'))
      console.error('Error sending chat query:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return { sendMessage, response, isLoading, error }
}