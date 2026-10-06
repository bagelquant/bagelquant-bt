# Transaction costs

Research weights use `cost_return = cost_rate * sum(abs(target - Net pretrade
 drift weight))`, including initial entry; default cost_rate is 0.0005. Net
subtracts this amount directly from its period return. Virtual research cash may
reflect fees without a capital/lot/minimum-fee constraint. Gross and Net holdings
drift independently; Net pre-cost return minus its cost equals Net return, while
independent Gross return need not satisfy that subtraction after drift.

Execution commission defaults to max(executed notional * 0.0005, 5) per order and
execution date. No fill means no fee. Separate IDs are charged separately; a
later residual retry is another daily fill. Defaults have no tax/slippage.
Caller inputs may declare buy/sell slippage, sell tax and transfer fee. Cash
includes all charges when selecting affordable whole shares.

Custom identified pure rules return a nonnegative fee and valid fill price;
quote quantity cannot decrease total buy charge. ID/version/parameters enter
numerical identity. Fee-free reference P&L, actual fill-price P&L and explicit
fees are separate; implementation shortfall includes unfilled shares and price
changes as well as fees. FIFO allocates entry/exit costs by matched quantity.

Native stress scales only the configured component. Minimum fees/tax stay fixed
for proportional commission stress; zero slippage stays zero. Opaque custom
rules require explicit caller-declared scenarios.
