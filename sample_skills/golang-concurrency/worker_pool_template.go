// Package workerpool is a production template for a bounded worker pool with
// graceful shutdown. Copy it and replace Job / Result / process.
package workerpool

import (
	"context"
	"errors"
	"runtime"
	"sync"
)

// Job is a unit of work.
type Job struct {
	ID      int
	Payload []byte
}

// Result is the outcome of processing a Job.
type Result struct {
	JobID int
	Err   error
}

// Run processes jobs with `workers` goroutines until jobs is closed or ctx is
// cancelled. It never leaks goroutines: all workers exit before Run returns
// the results channel is closed.
func Run(ctx context.Context, workers int, jobs <-chan Job, process func(context.Context, Job) error) <-chan Result {
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	results := make(chan Result)
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case job, ok := <-jobs:
					if !ok {
						return
					}
					err := process(ctx, job)
					select {
					case results <- Result{JobID: job.ID, Err: err}:
					case <-ctx.Done():
						return
					}
				}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()
	return results
}

// Collect drains results and joins all errors.
func Collect(results <-chan Result) error {
	var errs []error
	for r := range results {
		if r.Err != nil {
			errs = append(errs, r.Err)
		}
	}
	return errors.Join(errs...)
}
