# Compact BT metadata

BTStore metadata schema 2 keeps small descriptor headers in SQLite. Exact
coordinate counts and content evidence are immutable Zstd Parquet blocks in
`<artifact_path>/indexes/`. Each block contains at most 65,536 records and
8 MiB of decoded columnar data. Identical encoded blocks share one SHA-256 name.
Numerical files, table references, evaluation/chapter/shared receipts and their
identities retain their original meaning.

Descriptors read compact headers only. Date/range/exact-key selections prune
blocks before decoding and count exact matching coordinates; empty keys read
no blocks. Compositions retain ordered original references, predicates and
counts. They do not publish another expanded coordinate grid. Append and repair
reuse original tables and their index blocks without recomputing history.
Explicit maintenance preserves the evidence capabilities of retained references.

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

## Initialization and incompatible storage

Initialization, reads and writes require schema 2. Schema-1 converters and their
legacy index parser are retired; startup does not migrate storage. Initialize a
fresh store through the owning API. Replacing an existing operational store is
a separate, explicitly authorized action after all readers and writers stop.
Numerical history and receipt identities are not rewritten for compatibility.

## Read-only storage inventory

`BTStore.storage_usage(within_root=None)` reports metadata and WAL/SHM/journal
sidecars, numerical files, compact index blocks, hidden and unknown owned files.
The inventory includes lexical absolute paths, logical bytes, allocated bytes
(`st_blocks * 512`), file kind and device/inode. Unique-inode totals count hard
links once; symbolic links count their own object, never their targets.
`within_root` limits traversal to selected root paths. This API stats files and
never opens numerical bodies, initializes storage or runs recovery.
