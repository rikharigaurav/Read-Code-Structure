import { useState } from 'react';
import { submitGitHubRepo } from '@/lib/api/client';
import type { GitHubResponse } from '@/lib/types/api';

interface UseGitHubResult {
  submitRepo: (repoUrl: string) => Promise<void>;
  localPath: string | null;
  isLoading: boolean;
  error: Error | null;
}

export function useGitHub(): UseGitHubResult {
  const [localPath, setLocalPath] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const submitRepo = async (repoUrl: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await submitGitHubRepo({ repo_url: repoUrl });
      setLocalPath(result.response.localPath);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('An error occurred'));
    } finally {
      setIsLoading(false);
    }
  };

  return { submitRepo, localPath, isLoading, error };
}