# Compact BT metadata and offline conversion

BTStore metadata schema 2 keeps small descriptor headers in SQLite. Exact
coordinate counts and content evidence are immutable Zstd Parquet blocks in
`<artifact_path>/indexes/`. Each block contains at most 65,536 records and
8 MiB of decoded columnar data. Identical encoded blocks share one SHA-256 name.
Numerical files, table references, evaluation/chapter/shared receipts and their
identities retain their original meaning.

Schema 1 embedded an expanded JSON object for every coordinate and evidence
group in `bt_table_index.schema_json`. Repeated dates, asset names and property
names dominated that representation. Composed references could also retain
another expanded historical grid. This is live metadata, not necessarily free
SQLite pages; vacuum alone does not remove it.

Descriptors read compact headers only. Date/range/exact-key selections prune
blocks before decoding and count exact matching coordinates; empty keys read
no blocks. Compositions retain ordered original references, predicates and
counts. They do not publish another expanded coordinate grid. Append and repair
reuse original tables and their index blocks without recomputing history.
Explicit maintenance preserves migrated evidence capabilities.

Block reads verify encoded identity, typed schema, row count and pruning bounds.
An operation rechecks consumed file stats before returning, including memo hits.
Deep reference verification checks every original, coordinate and evidence
record and recomputes pruning bounds. Publication and orphan cleanup share the
SQLite writer fence. Cleanup retains blocks reachable from any retained header,
rechecks frozen candidates and records resumable receipts; unknown files,
directories, symlinks and corrupt blocks are not disposable candidates.

`read_context(cache_bytes=...)` reserves one quarter of the supplied cache for
metadata and the remainder for numerical frames. It never increases the caller's
budget. A block decode is capped independently of caching. Evidence combination
uses disposable disk-backed sorting with a 4 MiB page cache and streams canonical
daily digests, avoiding a Python dictionary of every historical coordinate.
The returned date catalogue and compact block headers still scale with their
number of dates/blocks.

## Conversion commands

Normal initialization, reads and writes require schema 2; startup never migrates.
Use the separately invoked owner module from an environment containing this
version of BT and Core. Paths below are placeholders for the selected store.

```bash
uv run --python 3.13 python -m bagelquant_bt.metadata_upgrade plan \
  --meta-path /selected/data/bt_meta.sqlite \
  --artifact-path /selected/data/production/calculations/bt
```

`plan` preserves source and sidecar bytes, using a private committed metadata
snapshot. It reports index JSON bytes, largest cell and a conservative free-space
estimate of four times the database size. This is staging headroom, not a predicted
compression ratio. A backup retained under the data root remains part of the
overview total after conversion.
JSON sizing reads incremental BLOB lengths without materializing giant cells.
The system temporary directory also needs room for the committed planning copy
and bounded parser spools; use a spacious `TMPDIR` for an operational conversion.

For an explicitly authorized cutover, stop **all** readers and writers of the
store, including Workbench workers, before applying. The confirmation flag is
the operator's assertion that they are stopped; it does not stop services.
Coordinate deployment of schema-2 source with this offline operation.

```bash
uv run --python 3.13 python -m bagelquant_bt.metadata_upgrade apply \
  --meta-path /selected/data/bt_meta.sqlite \
  --artifact-path /selected/data/production/calculations/bt \
  --backup-directory /selected/backups \
  --confirm-stopped
```

The converter:

1. Holds a conversion lock and a SQLite writer reservation. Captures committed
   main/WAL content with SQLite backup, runs integrity and logical-byte checks,
   and retains a checksummed `original.sqlite`.
2. Streams even a single large SQLite cell through 64 KiB reads. Bounded records
   and typed batches replace whole-cell `json.loads`. Temporary array fragments
   allow verification of the original canonical digest even with reordered JSON
   fields. Missing, null and empty capabilities remain distinct.
3. Copies receipt/manifests and reference JSON exactly. Reads no numerical
   Parquet and executes no producer, account or statistic. Reconstructs every
   converted metadata array to verify the exact original canonical contents.
   Before dropping a composed grid, verifies its counts against original indexed
   parts using a disposable disk-backed equality check. Missing original
   coordinate authority blocks conversion with a maintenance requirement.
4. Verifies and publishes immutable blocks first. Rechecks the stopped source,
   checkpoints WAL, obtains exclusive access and atomically replaces the main
   database last. The converted database uses WAL. It verifies the published
   headers and blocks before marking the operation complete.

The journal beside the source binds the backup, source fingerprint and exact
marked staging directory. Repeating the same command retries an interrupted
attempt; after process death, only that journal's verified staging directory is
discarded. Published blocks can be reused. A changed source, backup, conflicting
block, active reader/writer or invalid legacy digest prevents the switch.
New and retried directory entries are synced through their parents before the
backup or published blocks become database authority.
Preserve the journal and backups. A source change after an interrupted attempt
requires a separately reviewed recovery, rather than overwriting its evidence.

Rollback is explicit and allowed only if the converted logical database has
received no new writes:

```bash
uv run --python 3.13 python -m bagelquant_bt.metadata_upgrade rollback \
  --meta-path /selected/data/bt_meta.sqlite \
  --artifact-path /selected/data/production/calculations/bt \
  --confirm-stopped
```

Rollback verifies the original backup and a separately preserved converted-state
backup, then restores schema 1 atomically. New blocks remain as recovery evidence.
Rollback uses the same marked, journaled staging recovery; retry cleans its
abandoned copy even if the database replacement completed before interruption.
Restore the matching schema-1 application code before restarting. No command
deletes the preserved backups or automatically changes service state.

The implementation uses Python's [incremental SQLite BLOB and backup APIs](https://docs.python.org/3.13/library/sqlite3.html)
and explicitly handles SQLite's [WAL lifecycle](https://www.sqlite.org/wal.html).

## Read-only storage inventory

`BTStore.storage_usage(within_root=None)` reports metadata and WAL/SHM/journal
sidecars, numerical files, compact index blocks, hidden and unknown owned files.
The inventory includes lexical absolute paths, logical bytes, allocated bytes
(`st_blocks * 512`), file kind and device/inode. Unique-inode totals count hard
links once; symbolic links count their own object, never their targets.
`within_root` limits traversal to selected root paths. This API stats files and
never opens numerical bodies, initializes storage or runs recovery.
