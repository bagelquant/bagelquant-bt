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

Window/lag statistics reuse prepared rank/return primitives rather than simulate
accounts. Saved-period reads aggregate results and select the correct historical
lot snapshot; they never rebuild values or replay execution.

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

Chapter inventory can use metadata-only receipts instead of opening every saved period. Selected-table reads hash each distinct selected file once per finite context, then recheck its identity after decoding. Repeated selected reads share proofs; unrelated period/table corruption remains visible to full audits and reads consuming it. Evaluation formulas, causal execution and result identity are unchanged.
