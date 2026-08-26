import { afterEach, describe, expect, it } from 'vitest'

import { type MockServer, restartMockServer, startMockServer } from '../e2e/mock-server'

interface MockCompletion {
  choices: Array<{
    message: {
      tool_calls?: Array<{ function?: { name?: string } }>
    }
  }>
}

const servers: MockServer[] = []

async function firstCrossTurn(server: MockServer): Promise<MockCompletion> {
  const response = await fetch(`${server.url}/v1/chat/completions`, {
    body: JSON.stringify({
      messages: [{ role: 'user', content: 'E2E_SIDEBAR_CROSS' }],
      model: 'mock-model',
      stream: false
    }),
    headers: { 'content-type': 'application/json' },
    method: 'POST'
  })

  expect(response.ok).toBe(true)
  return (await response.json()) as MockCompletion
}

function toolNames(completion: MockCompletion): string[] {
  return completion.choices[0]?.message.tool_calls?.map(call => call.function?.name ?? '') ?? []
}

afterEach(async () => {
  await Promise.all(servers.splice(0).map(server => server.close()))
  restartMockServer()
})

describe('mock inference server script isolation', () => {
  it('starts each concurrent server at the first sidebar-cross turn', async () => {
    restartMockServer()
    servers.push(await startMockServer(), await startMockServer())

    const completions = await Promise.all(servers.map(firstCrossTurn))

    for (const completion of completions) {
      expect(toolNames(completion)).toEqual(['terminal', 'delegate_task'])
    }
  })
})
