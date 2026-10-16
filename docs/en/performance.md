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

Research weight paths retain interval and availability scalars during the daily
loop and materialize the returned weight checkpoint once, after the final mature
session. Asset and rebalance sets are reduced to unique coordinates before
Python conversion. These allocation reductions preserve daily outputs, zero
asset coordinates and continuation state.

Window/lag statistics reuse prepared rank/return primitives rather than simulate
accounts. Saved-period reads aggregate results and select the correct historical
lot snapshot; they never rebuild values or replay execution.

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
