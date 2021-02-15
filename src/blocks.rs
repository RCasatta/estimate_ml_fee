use crate::transactions::Transactions;
use bitcoincore_rpc::bitcoin::Block;
use std::collections::{HashMap, VecDeque};

#[derive(Debug)]
pub struct BlocksBuckets {
    last_blocks: VecDeque<Block>, //TODO may use (height, block) and check last 10
    buckets: Option<Vec<u64>>,
    buckets_limits: Vec<f64>,
    blocks_to_consider: usize,
}

pub fn create_buckets_limits(increment_percent: u32, upper_limit: f64) -> Vec<f64> {
    let mut buckets_limits = vec![];
    let increment_percent = 1.0f64 + (increment_percent as f64 / 100.0f64);
    let mut current_value = 1.0f64;
    loop {
        if current_value >= upper_limit {
            break;
        }
        current_value *= increment_percent;
        buckets_limits.push(current_value);
    }
    buckets_limits
}

impl BlocksBuckets {
    pub fn new(increment_percent: u32, upper_limit: f64, blocks_to_consider: usize) -> Self {
        let buckets_limits = create_buckets_limits(increment_percent, upper_limit);
        let buckets = None;
        Self {
            last_blocks: VecDeque::new(),
            buckets_limits,
            blocks_to_consider,
            buckets,
        }
    }

    fn full(&self) -> bool {
        self.blocks_to_consider == self.last_blocks.len()
    }

    pub fn add(&mut self, block: Block) {
        if self.full() {
            self.last_blocks.pop_back();
        }
        self.last_blocks.push_front(block);
        if self.full() {
            let mut map = HashMap::new();
            for b in self.last_blocks.iter() {
                for tx in b.txdata.iter() {
                    map.insert(tx.txid(), tx.clone());
                }
            }
            let txs = Transactions::from_txs(map);
            let rates = txs.fee_rates();
            let mut buckets = vec![0u64; self.buckets_limits.len()];
            for rate in rates {
                let index = self
                    .buckets_limits
                    .iter()
                    .position(|e| e > &rate)
                    .unwrap_or(self.buckets_limits.len() - 1);
                buckets[index] += 1;
            }

            self.buckets = Some(buckets);
        }
    }

    pub fn get_buckets(&self) -> &Option<Vec<u64>> {
        &self.buckets
    }
}

#[cfg(test)]
mod tests {
    use crate::blocks::BlocksTimes;
    use bitcoincore_rpc::bitcoin::blockdata::constants::genesis_block;
    use bitcoincore_rpc::bitcoin::Network;

    #[test]
    fn test_block_times() {
        let mut bt = BlocksTimes::new();
        assert_eq!(bt.time(0), 0, "giving time even if there aren't");
        let b1 = genesis_block(Network::Bitcoin);
        bt.add(1, &b1);
        assert_eq!(bt.time(1), 0, "considering empty block");
        let mut b2 = b1.clone();
        b2.txdata.push(b1.txdata[0].clone());
        b2.header.time = 2;
        bt.add(2, &b2);
        assert_eq!(bt.time(2), 2, "getting wrong time");
        let mut b3 = b1.clone();
        b3.header.time = 3;
        bt.add(3, &b3);
        assert_eq!(bt.time(3), 2, "getting wrong time because last is empty");
        let mut b4 = b2.clone();
        b4.header.time = 4;
        bt.add(4, &b4);
        assert_eq!(bt.time(4), 4, "getting non last time");
    }
}
