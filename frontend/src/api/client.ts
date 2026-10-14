import type { QueryResponse } from './types'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function askQuestion(
  question: string,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  const response = await fetch('/api/queries', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question }),
    signal,
  })
  if (!response.ok) {
    throw new ApiError(response.status, `request failed with status ${response.status}`)
  }
  return (await response.json()) as QueryResponse
}
