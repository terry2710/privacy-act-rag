import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, askQuestion } from './client'

const sample = {
  request_id: 'req123',
  session_id: 'sess1',
  answer: 'An entity must handle personal information openly.',
  citations: [],
  outcome: 'answered',
  pipeline_version: 'fixed-dense-v1',
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('askQuestion', () => {
  it('posts the question and returns the parsed response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(sample), { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const result = await askQuestion('What is APP 1?')

    expect(result.request_id).toBe('req123')
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/queries')
    expect(JSON.parse(init.body)).toEqual({ question: 'What is APP 1?' })
  })

  it('throws ApiError with the status on a failed response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 502 })))

    await expect(askQuestion('What is APP 1?')).rejects.toMatchObject({
      name: 'Error',
      status: 502,
    })
    await expect(askQuestion('What is APP 1?')).rejects.toBeInstanceOf(ApiError)
  })
})
