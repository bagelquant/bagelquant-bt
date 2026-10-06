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
