---
name: golang-concurrency
description: Worker pools, errgroup, context cancellation, bounded parallelism, and channel pipelines in Go
tags: [golang, go, concurrency, goroutines, errgroup, channels, context]
scope: global
---

# Go Concurrency Patterns

Conventions for writing concurrent Go services that are leak-free, cancellable and
observable. Prefer structured concurrency: every goroutine must have an owner that
waits for it and a context that can stop it.

## Golden Rules

- Never start a goroutine without knowing how it will stop. Every `go func()` needs an
  exit path driven by `ctx.Done()`, a closed channel, or a finite input.
- The function that starts goroutines is responsible for waiting on them
  (`sync.WaitGroup` or `errgroup.Group`). Do not fire-and-forget in request handlers.
- Pass `context.Context` as the first parameter. Never store contexts in structs.
- Prefer `errgroup` over hand-rolled `WaitGroup` + error channels.
- Channels are for ownership transfer and signalling; protect shared state with
  `sync.Mutex` when that is simpler.

## errgroup with Context Cancellation

Use `golang.org/x/sync/errgroup` to fan out work. `errgroup.WithContext` returns a derived
context that is cancelled as soon as any goroutine returns a non-nil error, so sibling
goroutines stop early instead of burning CPU on doomed work.

```go
g, ctx := errgroup.WithContext(ctx)
g.SetLimit(8) // bounded parallelism, available since x/sync v0.1.0

for _, id := range ids {
    id := id
    g.Go(func() error {
        return fetchAndStore(ctx, id)
    })
}
if err := g.Wait(); err != nil {
    return fmt.Errorf("sync batch: %w", err)
}
```

`g.SetLimit(n)` replaces the old semaphore-channel idiom. `g.TryGo` lets you shed load
instead of blocking when the limit is reached.

## Worker Pool

For long-lived pipelines with a stream of jobs, use a fixed worker pool reading from a
jobs channel. See `worker_pool_template.go` in this skill for a production template with
graceful shutdown and result collection.

- Size the pool from configuration, default to `runtime.GOMAXPROCS(0)` for CPU-bound work
  and a larger number (e.g. 32-64) for I/O-bound work.
- The producer closes the jobs channel; workers range over it.
- Close the results channel only after `wg.Wait()` returns, in a separate goroutine.

## Preventing Goroutine Leaks

Leaks happen when a goroutine blocks forever on a send, receive, or a call that never
returns. The classic case is calling a slow dependency without a deadline: when the remote
service stops responding, every request piles up another blocked goroutine until the
process runs out of memory.

- Always derive a timeout for outbound calls: `ctx, cancel := context.WithTimeout(ctx, 2*time.Second)`
  and `defer cancel()`.
- Use `http.NewRequestWithContext` and set `http.Client{Timeout: ...}`; the zero-value client has no timeout.
- When sending on a channel, `select` on `ctx.Done()` as well so the sender can bail out.
- Use buffered channels of size 1 for "first result wins" patterns so losers do not block.
- Detect leaks in tests with `go.uber.org/goleak`: `defer goleak.VerifyNone(t)`.

```go
select {
case out <- result:
case <-ctx.Done():
    return ctx.Err()
}
```

## Channel Pipelines

Build pipelines as stages `func(ctx context.Context, in <-chan T) <-chan U`. Each stage
owns and closes its output channel. Fan-out by starting N copies of a stage, fan-in with
a merge function that uses a `WaitGroup` to close the merged channel.

### Backpressure

Unbuffered channels give natural backpressure. Add buffers only to absorb measured bursts,
never to "fix" a deadlock.

## Testing and Race Detection

- Run `go test -race ./...` in CI; data races are bugs even when tests pass.
- Avoid `time.Sleep` in tests; synchronise with channels or use a fake clock.
- Use `t.Cleanup` to cancel contexts and stop workers started by a test.
