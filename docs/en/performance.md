# Local execution and resources

`BTExecutionOptions(workers=..., limits=...)` accepts a caller-supplied Core
ResourceLimits budget. `run_evaluation_batch(items, evaluate, options=...)` bounds
independent in-flight evaluations, divides the shared budget, retains input order
and stops admission on cancellation/failure. BT never probes CPU/RAM or chooses
a machine-wide policy. Numerical identity excludes workers, memory and batches.

An individual account remains sequential because cash, settlement, FIFO and
later decisions depend on actual prior fills. Its output uses bounded columnar
buffers and cancellation checks each session/order/planning step. Callers may
share read-only market inputs, while every scenario owns independent account
state. Reuse verified saved artifacts before preparing market rows.

Execution and saved-weight scheduling build a date-to-calendar-index mapping
once per operation. Session and rebalance lookup therefore takes linear total
calendar work rather than scanning the full calendar for every decision. This
does not change sizing, order, settlement, checkpoint identity or sequential
account state. Append retains the existing historical checkpoint and processes
only its admitted new sessions.

Research weight paths retain interval and availability scalars during the daily
loop and materialize the returned weight checkpoint once, after the final mature
session. Asset and rebalance sets are reduced to unique coordinates before
Python conversion. These allocation reductions preserve daily outputs, zero
asset coordinates and continuation state.

Alpha's default diagnostics include 87 independent causal weight paths: Book,
Spread, 61 Book lead/lag paths, 14 Alpha-return lag paths and ten quantiles.
They retain independent Gross/Net holdings, cost and continuation state. The
engine reduces date/asset coordinates before Python conversion and materializes
only its final mature checkpoint. An evaluation can share immutable sparse
return indices, null-filled returns, observation masks and interval scalars
across paths with the exact asset axis. Different date windows select from that
preparation only when the common source has no additional assets. Each path
retains its own daily vectors, costs, Gross/Net holdings and checkpoints, with
the same arithmetic order. Holdings output still records every requested
processed session. No historical account state is recreated during append.

This evaluation-local LRU uses at most the minimum of `cache_mib` and one quarter
of `memory_target_mib`. Its reserve is deducted from both the existing local
cache and memory allowances before other diagnostics are admitted. It can use
the full configured cache for repeated return panels, leaving zero ordinary
cache when that is the binding limit. There is no additional memory budget
or machine-dependent policy. Admission
precedes partition/NumPy allocation; zero, tiny or oversized cases use the
ordinary per-path preparation without retaining a shared source frame or asset
set. A common append source starts after the earliest saved path checkpoint.
The cache clears on success, failure or cancellation; it stores no account
state or persisted validity. Execution reports expose `return_preparation_hits`,
`return_prepared_rows`, `return_preparation_peak_bytes` and
`return_preparation_maximum_bytes`. These charges are conservative working-set
estimates, not process RSS bounds. Components without causal paths reserve no
return preparation.

Quantile-curve preparation groups numerically ordered quantile returns in one
columnar query before applying the existing NumPy/SciPy formulas to each curve.
Sparse, incomplete and constant curves retain their original unavailable reasons;
rank ties, correlation, covariance and rounding are unchanged.
Empty horizon primitives expose the same count columns and dtypes as populated
ones, so continuation from a short history cannot promote later count schemas.
Continuation also strictly normalizes retained legacy count columns at the
current primitive merge boundary. An empty retained frame uses that primitive's
current empty schema. Existing saved tables, receipt identities and processed
daily checkpoints remain unchanged; no historical path replay is needed.
Continuation publication requires equal schemas as well as equal values before
reusing an original reference. A changed derived dtype is published through the
ordinary immutable table path, including an empty result; reload preserves the
current typed boundary without rewriting the original artifact.

Saved-period reads aggregate results and select the correct historical lot
snapshot; they never rebuild values or replay execution. Complete Alpha
benchmarks must include all default horizon, persistence and causal path work;
a persistence-only benchmark does not establish full-evaluation latency.

Signal persistence shares validated finite factors and session coordinates with
Alpha evaluation. Each bounded pair chunk collects its join, common-universe
ranks and correlation together; it retains the existing session and row bounds.
Append calculates only new target pairs and summarizes once after merging the
saved primitives. Return-label validation shares one per-date interval summary
for consistency and overlap checks. These query boundaries do not change rank
ties, missing-value treatment, formulas or native pool allocation.

Tests use synthetic inputs and isolated roots. Broad benchmarks must preserve
causal, deterministic and typed numerical contracts and report actual measured
limits; runtime pressure never changes mathematics.

Rank-persistence derived summaries use `bt.persistence.canonical_mean.v1` and
the public stdlib `statistics.mean` finite reducer. This removes native
floating-reduction scheduling from the summary policy, but its Python/exact-ratio
cost is not yet measured here. No whole-Evaluation latency or memory improvement
is claimed. A pre-policy same-boundary continuation reduces the retained daily
correlations once, without rerunning correlations or weight paths; subsequent
current-policy repeats retain the summary object and skip that reduction.
The direct evaluator's eager supplied-input validation remains unchanged.
Full scientific comparisons must name this last-bit policy change explicitly
and preserve the original failed/legacy results; do not replace an exact-bit
comparison with a floating tolerance to conceal the old reduction instability.
Public mean output dtypes are inferred from lazy schema metadata only. Float32
retains its final output cast; integer/Boolean/Decimal inputs use the same
canonical numeric mean with Float64 output. No legacy data reduction runs to
infer types, and String/Null inputs retain unavailable values without numeric
coercion. Decimal adaptation uses exact public Fraction values before stdlib
mean, independent of the ambient Decimal precision.

`evaluate_alpha(..., options=..., progress=...)` runs independent horizon
windows through that same pool. It prepares factor, labels, Book and Spread once,
then merges private window outputs in declared order before global inference.
Omitted options and panels with fewer than 1024 factor rows remain serial.
Progress reports `(completed, total)` windows, starting at zero, on the caller
thread. Cancellation checks also run between window stages and summaries.

Worker admission reserves 20% of the caller memory budget plus the estimated
size of shared frames. Each window estimates at least 64 MiB or eight times
factor-plus-label bytes, whichever is larger. This reduces requested concurrency
to fit the estimate, with one worker as the minimum; these estimates are not
a hard allocation limit. `EvaluationResult.execution` reports requested/actual
workers and byte estimates. These values are execution evidence only and must
not enter numerical metrics or result identity. Callers use either outer target
parallelism or inner window parallelism to avoid nested pools.

Chapter inventory can use metadata-only receipts instead of opening every saved period. Selected-table reads trust registered descriptors, row counts and indexed schemas, then recheck file identity after decoding. Explicit verify owns byte/logical/index audits. Repeated selected reads share proofs; unrelated period/table corruption remains visible to full audits and reads consuming it. Evaluation formulas, causal execution and result identity are unchanged.


A caller may supply `BTExecutionOptions(native_admission=...)`, an execution-only
callable returning a context manager for a native-work class. BT evaluates its
complete mixed unit inside that permit and uses one local worker, retaining the
whole supplied memory/thread share. This prevents nested native executors when
Polars shares a process pool with BLAS/OpenMP. BT does not configure or resize
those pools, detect hardware or choose machine policy. Omitted admission keeps
the standalone bounded thread pool. Its ready slots refill after any completed
future; results retain input order and failure stops further admission.


## Publication lookup work

Shared and chapter publication resolve a batch of new table IDs with one
historical-receipt pass, rather than a separate scan for each table. Lookup work
therefore grows with historical metadata plus the output batch. It is still
history-dependent: publishing one new table is not a constant-time operation
when its descriptor is absent. Indexed hits avoid historical lookup.

Admitted delta continuation uses the same batch resolver before composing
original prefix/suffix references. It validates every output interval first,
hashes each supplied frame once (sharing an identical frame object's hash),
and resolves all missing new IDs together. An unchanged receipt frontier needs
one history pass for the whole batch. Concurrent publication can require a
fresh pass; missing optional indexes still cannot authorize recreating a lost
original. `BTStore(publication_observer=...)` optionally reports inclusive
`hashing`, `lookup`, `indexing` and `composition` seconds, including failures.
These nested observations must not be summed as exclusive elapsed time.

The batch retains only requested matches. Original descriptors, nested
compositions and receipt authentication remain authoritative; a missing optional
index cannot authorize recreating a lost original. Fresh indexed checks and a
receipt frontier catch intervening owner publications. Owner receipt rows are
append-only; invalidation changes status and cleanup retains receipt history.
On interruption, lookup releases its own raw/decoded receipt and traversal
aliases without altering another frame's exception state. Formulas, table and
receipt identities, schemas and native resource limits are unchanged.

A controlled local comparison on Python 3.13 with six Polars threads used three
paired trials per case and identical one-row outputs. With 64 MiB of unrelated
chapter receipts, 32-table shared publication changed from 3.15 s to 0.21 s;
chapter publication changed from 3.15 s to 0.20 s. Table and receipt identities
matched. At zero history, the extra fresh checks added about 0.5 ms per new
table. These are publication measurements, not a complete Alpha latency promise.

An additional Python 3.13/six-Polars-thread continuation comparison used three
paired trials, 277 tiny delta tables and 64 MiB of unrelated receipt metadata.
Median publication changed from 32.98 s to 3.38 s; history passes changed from
277 to one. Table/composition references and values matched exactly. At zero
history with 32 tables, medians were 0.35 s and 0.38 s. These component probes
vary metadata separately from numerical input size and do not certify a
production five-minute evaluation target.

## Isolated horizon workers

`BTExecutionOptions.window_dispatch` accepts a caller implementing
`HorizonDispatch`. BT provides complete child workspace and coordinator retained
input/output estimates before staging. The caller may reduce admission to one,
which uses the ordinary inline path without staged transport. An admitted batch
stages shared inputs once and supplies immutable `HorizonWork` descriptors.
`compute_horizon_work` calls the same window kernel, writes private bounded
outputs and returns `HorizonResult`; it never opens a backend store.

The caller drains parent native work before children run and restores the parent
permit before BT validates and reads outputs. Results merge in requested window
order before inference/reductions. Actual worker counts and CPU seconds remain
execution evidence, outside mathematical identities. Cancellation must reap
children before temporary staging disappears. Same-boundary reuse bypasses the
dispatcher. Causal weight/account paths retain their sequential state.

On an isolated six-thread/16-GiB synthetic full-Alpha workload with 4,096
sessions and 1,024 assets, three alternating paired trials gave median kernel
times of 108.82 s inline and 104.36 s with two three-thread children. All table
hashes and state matched exactly; two children were active. The same comparison
at 1,024 sessions and 512 assets was effectively neutral (12.12/12.13 s).
Workbench therefore keeps Auto inline below a 1 GiB estimated child workspace,
including its interpreter reserve; descriptor and coordinator sizing can reduce
admission further. Child inputs use immutable mapped Arrow buffers, while parent
outputs are copied before staging expires. This kernel probe includes all 87
causal paths but excludes Workbench preparation, chapters and report export.

## Call-local table-index decoding

An ordinary `BTStore.read_reference` shares the canonical registered-reference
validator with `describe_reference`. T1 checks the registered descriptor,
original-column metadata and complete index digest. Its read-only transaction
closes before selection, frame-cache lookup or Parquet work. The raw reference
and schema serialization and the parsed index belong only to that read call;
there is no persistent metadata memo or new identity/schema/policy.

T2 still opens a fresh read transaction before selection or a leaf frame-cache
hit. It fetches both `reference_json` and `schema_json` and requires exact equality
with the serialization validated in T1 before reusing the parsed object. A
physical leaf read opens another fresh transaction at T3, makes the same exact
comparison and retains the physical typed-schema check. Appearance, disappearance
or any serialization change fails with `BT table index changed during read`.
Thus a whitespace-only maintenance change can stop an overlapping read; the next
call validates the new serialization normally. The reader neither rewrites rows
nor holds a long snapshot that hides index maintenance. A metadata change after
the last observation, or an unobserved change-and-restore, is not newly detected.
File-stat invalidation, row counts, coordinate pruning, empty typed selections,
concat leaf order and explicit byte/logical audit behavior remain unchanged.
Registered index-free history still supports unfiltered reads and requires
explicit maintenance for selected reads.

The improvement removes duplicate JSON decoding within an unchanged indexed
read. It retains the fresh SQL queries and serialization comparisons. Stored
schema-JSON length is not per-job bytes decoded; source inspection alone does
not establish elapsed benefit. Owner and downstream tests and bounded isolated
measurements must supply that evidence before a performance claim.

Call-local lifetime is finite, not a hard memory allowance. Let R and S denote
the Python storage of the raw reference and schema strings, P the parsed tree,
and D/U the existing canonical digest's temporary JSON string/UTF-8 bytes. A
leaf read's metadata peak includes P plus the larger of R+S+D+U (T1 validation)
and 2(R+S) (T2/T3 fresh serialization), plus SQLite/allocator overhead. Physical
frames and the existing 64 MiB decoded-frame memo are additional. For a concat,
the parent record stays live while one flat leaf is read, so both records count;
there is no accumulation of parsed leaf records across iterations. Python Unicode
storage can exceed UTF-8 file length. This estimate makes no hard RSS or resource
admission claim and introduces no extra cross-call budget/cache.

Every exit clears the new record's own fields and reader-owned index/fact aliases;
lookup and fresh-query rows are released in `finally`. It does not mutate a
borrowed descriptor/index tree, result, cached frame or caller traceback. Existing
JSON exceptions may retain their input document, and unchanged called functions
or foreign traceback frames can retain arguments. Those pre-existing exception
retention limits remain; no blanket release guarantee or error rewriting is used.
