# Bitcoin fee estimation with machine learning

Estimate the fee_rate needed in `[sat/vbyte]` to enter the blockchain waiting `block-target` blocks

## Requirements

Get needed informations from a running local bitcoin core node (pruned nodes are ok)

## Example

```
$ cargo run --release -- --blocks-target 30 --cookie-path $HOME/.bitcoin/.cookie --rpc-address http://127.0.0.1:8332 --model-path 20210111-164919-model
...
[2021-01-13T14:56:17Z INFO  estimate_ml_fee] all mempool txs 8265, not old: 3807
[2021-01-13T14:56:18Z INFO  estimate_ml_fee] mempool_txid:3807 of which with input in last 6 blocks:348 (9.1%)
[2021-01-13T14:56:18Z INFO  estimate_ml_fee] mempool buckets [1, 2, 3, 4, 4, 15, 11, 65, 29, 62, 16, 37, 73, 25, 0, 1]
[2021-01-13T14:56:18Z INFO  estimate_ml_fee] Estimated fee to enter in 30 blocks is 44.309048 sat/vbyte
```