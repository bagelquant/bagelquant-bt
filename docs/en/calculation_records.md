# Calculation records

Ordinary reads trust registered immutable metadata and consume typed values without byte/content rehashing. Explicit audits retain original integrity checks and compare present current-version derived indexes to original evidence. New outputs retain canonical hashes.

Public owner APIs: `BTStore.describe_reference / index_plan / build_index / verify`. Index maintenance is explicit, frozen-plan based and cancelable; normal opens never backfill or rewrite historical receipts/manifests. Missing/partial/version-mismatched derived indexes provide no selection/interval proof. Supported authoritative metadata exact hits and covering-parent shortcuts remain valid. Admission limits do not change numerical identity.
