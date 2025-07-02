// API Request Types
interface ChatResponseFields {
  summary: string
  procedural_knowledge?: string
  code_solution?: string
  visualization_query?: string
}

export interface ChatResponse {
  fields: ChatResponseFields
}

export interface ChatRequest {
  query: string
  context?: Record<string, any>
}

export interface GitHubRequest {
  repo_url: string;
}

// API Response Types
export interface ApiResponse<T> {
  status_code: number;
  response: T;
}


export interface GitHubResponse {
  localPath: string;
}