# Bitcoin fee estimation with machine learning

Estimate the fee_rate needed in `[sat/vbyte]` to enter in a range of target

## Requirements

Get needed informations from a running local bitcoin core node (pruned nodes are ok)

## Example

```
$ cargo run --release -- --cookie-path $HOME/.bitcoin/.cookie --rpc-address http://127.0.0.1:8332
[2021-02-17T18:58:57Z INFO  estimate_ml_fee] start
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] Blocks asked to node
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] load elapsed: 1.473515ms
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] i:1 f:153.48785
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] i:3 f:120.43874
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] i:6 f:108.100815
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] i:36 f:23.592606
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] i:72 f:16.325903
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] i:144 f:12.416816
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] i:432 f:8.3061905
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] i:1008 f:5.199027
[2021-02-17T18:58:58Z INFO  estimate_ml_fee] estimate elapsed: 1.431968ms
```