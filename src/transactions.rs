use bitcoincore_rpc::bitcoin::{Transaction, Txid};
use std::collections::HashMap;

pub struct Transactions {
    txs: HashMap<Txid, Transaction>,
    txs_output_values: HashMap<Txid, OutputValues>,
}

pub type OutputValues = Box<[u64]>;

impl Transactions {
    pub fn from_txs(txs: HashMap<Txid, Transaction>) -> Self {
        let mut txs_output_values: HashMap<Txid, OutputValues> = HashMap::new();
        for (txid, tx) in txs.iter() {
            let output_values: Vec<_> = tx.output.iter().map(|e| e.value).collect();
            txs_output_values.insert(*txid, output_values.into_boxed_slice());
        }
        Transactions {
            txs,
            txs_output_values,
        }
    }

    // fee rate in sat/vbytes
    pub fn fee_rate(&self, txid: &Txid) -> Option<f64> {
        let tx = self.txs.get(txid)?;
        let fee = self.absolute_fee(tx)?;
        Some((fee as f64) / (tx.get_weight() as f64 / 4.0))
    }

    pub fn fee_rates(&self) -> Vec<f64> {
        self.txs.keys().filter_map(|tx| self.fee_rate(tx)).collect()
    }

    fn absolute_fee(&self, tx: &Transaction) -> Option<u64> {
        let sum_outputs: u64 = tx.output.iter().map(|o| o.value).sum();
        let mut sum_inputs: u64 = 0;
        for input in tx.input.iter() {
            let outputs_values = self.txs_output_values.get(&input.previous_output.txid)?;
            sum_inputs += outputs_values[input.previous_output.vout as usize];
        }
        Some(sum_inputs - sum_outputs)
    }
}
