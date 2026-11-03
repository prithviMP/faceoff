---
type: Attested Computation
title: Refund total
description: Net refunds by sale ID
status: stable
verified:
- by: human:fictional-finance
  at: '2026-01-01T00:00:00Z'
runtime: sqlite
database: ../ledger.sqlite
parameters:
- name: sale_id
  type: string
  required: true
executor:
  resource: faceoff_lab.receipts.execute
attester:
  resource: faceoff_lab.receipts.inspect
---
# Definition
Include partial refunds, subtract reversals.

# Computation
```sql
SELECT SUM(CASE WHEN reversed = 1 THEN -amount_paise ELSE amount_paise END) FROM refunds WHERE sale_id = :sale_id
```
