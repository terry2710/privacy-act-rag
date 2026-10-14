export type Outcome = 'answered' | 'insufficient_evidence' | 'blocked'

export interface Citation {
  rank: number
  chunk_id: string
  page: number | null
  source: string | null
  score: number
  text: string | null
}

export interface QueryResponse {
  request_id: string
  session_id: string
  answer: string
  citations: Citation[]
  outcome: Outcome
  pipeline_version: string
}
