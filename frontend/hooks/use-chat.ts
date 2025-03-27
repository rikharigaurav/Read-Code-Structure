import { useState } from 'react';
import { sendChatQuery } from '@/lib/api/client';
import type { ChatRequest, ChatResponse, ApiResponse } from '@/lib/types/api';

interface UseChatResult {
  sendMessage: (query: string, context?: ChatRequest['context']) => Promise<void>;
  response: ChatResponse | null;
  isLoading: boolean;
  error: Error | null;
}

export function useChat(): UseChatResult {
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const sendMessage = async (query: string, context?: ChatRequest['context']) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await sendChatQuery({ query, context });
      setResponse(result.response);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('An error occurred'));
    } finally {
      setIsLoading(false);
    }
  };

  return { sendMessage, response, isLoading, error };
}