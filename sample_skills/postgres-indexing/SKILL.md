---
name: postgres-indexing
description: PostgreSQL index design, EXPLAIN ANALYZE query tuning, and zero-downtime index migrations with CREATE INDEX CONCURRENTLY
tags: [postgres, postgresql, sql, indexing, performance, explain, migrations, database]
scope: global
---

# PostgreSQL Indexing & Query Performance

How we design, verify and ship indexes in PostgreSQL 14+. Every new index must be
justified by a query plan and created without locking production writes.

## Diagnose Before Indexing

- Find slow statements with `pg_stat_statements` ordered by `total_exec_time`.
- Run `EXPLAIN (ANALYZE, BUFFERS)` on the real query with realistic parameters.
- Look for `Seq Scan` on large relations, `Rows Removed by Filter` in the millions, and big
  gaps between estimated and actual row counts (stale statistics: run `ANALYZE`).
- A nested loop over two big relations with no usable index on the inner side is the most
  common cause of multi-second joins. Index the foreign key column used in the join
  condition; Postgres does not create indexes on foreign keys automatically.

## Choosing the Right Index

### B-tree (default)

Equality and range predicates, `ORDER BY`, and joins. Column order in a composite index
matters: put equality columns first, then range / sort columns.

```sql
-- WHERE tenant_id = $1 AND created_at >= $2 ORDER BY created_at DESC
CREATE INDEX CONCURRENTLY idx_orders_tenant_created
    ON orders (tenant_id, created_at DESC);
```

### Covering and Partial Indexes

- `INCLUDE (col)` enables index-only scans without bloating the key.
- Partial indexes (`WHERE status = 'pending'`) keep hot-subset indexes tiny.

### GIN, GiST and BRIN

- GIN for `jsonb` containment (`@>`), arrays, and full-text `tsvector` search.
- GiST for ranges, geometry and exclusion constraints.
- BRIN for append-only, naturally ordered tables (time-series) — tiny and cheap.

## Zero-Downtime Index Migrations

- Always use `CREATE INDEX CONCURRENTLY` on tables that receive writes. A plain
  `CREATE INDEX` takes a `SHARE` lock that blocks inserts, updates and deletes for the
  whole build.
- `CONCURRENTLY` cannot run inside a transaction block: disable the migration framework's
  transaction wrapper for that migration (e.g. `atomic = False` in Django,
  `disable_ddl_transaction!` in Rails).
- A failed concurrent build leaves an `INVALID` index; detect it via `pg_index.indisvalid`
  and `DROP INDEX CONCURRENTLY` before retrying.
- Set `lock_timeout` (e.g. `SET lock_timeout = '5s'`) so migrations fail fast instead of
  queueing behind long transactions.
- Use `DROP INDEX CONCURRENTLY` to remove unused indexes found via `pg_stat_user_indexes.idx_scan = 0`.

## Join and Query Tuning

- Make join keys the same data type on both sides; implicit casts disable index usage.
- Increase `work_mem` per session for large hash joins and sorts that spill to disk.
- Replace `OFFSET` pagination with keyset pagination (`WHERE (created_at, id) < ($1, $2)`).
- Raise `default_statistics_target` or create extended statistics for correlated columns.

## Maintenance

- Monitor index bloat; rebuild with `REINDEX INDEX CONCURRENTLY` (PG 12+).
- Every index slows writes and consumes cache — drop indexes that are not used.
