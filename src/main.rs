use bitcoincore_rpc::bitcoin::Txid;
use bitcoincore_rpc::{Auth, Client, RpcApi};
use chrono::{DateTime, Datelike, Timelike, Utc};
use log::info;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::iter::FromIterator;
use structopt::StructOpt;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

#[derive(StructOpt, Debug)]
pub struct EstimateOptions {
    /// the number of blocks I am ok to wait for
    #[structopt(long)]
    pub blocks: u16,

    #[structopt(flatten)]
    pub node_config: NodeConfig,
}

#[derive(StructOpt, Debug)]
pub struct NodeConfig {
    /// Rpc address eg. "http://127.0.0.1:18332"
    #[structopt(long)]
    pub rpc_address: String,

    /// Path of the bitcoin cookie file
    #[structopt(long)]
    pub cookie_path: String,
}

impl NodeConfig {
    pub fn make_rpc_client(&self) -> Result<Client, Box<dyn Error>> {
        let auth = Auth::CookieFile((&self.cookie_path).into());
        let url = self.rpc_address.to_string();
        Ok(Client::new(url, auth)?)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init_from_env(
        env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "info"),
    );
    info!("start");
    let options = EstimateOptions::from_args();

    let mut model_inputs = ModelInputs::default();
    let utc: DateTime<Utc> = Utc::now();
    model_inputs.ints.insert(
        "day_of_week".to_string(),
        utc.weekday().num_days_from_monday() as i64,
    );
    model_inputs
        .ints
        .insert("hour".to_string(), utc.hour() as i64);
    model_inputs
        .floats
        .insert("confirms_in".to_string(), options.blocks as f32);

    call_node(&mut model_inputs, &options.node_config)?;

    call_model(model_inputs)?;

    Ok(())
}

#[derive(Default)]
struct ModelInputs {
    pub floats: HashMap<String, f32>,
    pub ints: HashMap<String, i64>,
}

fn call_node(data: &mut ModelInputs, node_config: &NodeConfig) -> Result<(), Box<dyn Error>> {
    let client = node_config.make_rpc_client()?;

    let mut blocks_to_ask = vec![];
    let best_block_hash = client.get_best_block_hash()?;
    let mut last = best_block_hash;
    for _ in 0..6 {
        blocks_to_ask.push(last);
        let block_info = client.get_block_info(&last)?;
        last = block_info.previousblockhash.expect("found a stale block");
    }

    info!("asking blocks {:?}", blocks_to_ask);
    let mut txs = HashMap::new();
    for hash in blocks_to_ask {
        let block = client.get_block(&hash)?;
        for tx in block.txdata {
            txs.insert(tx.txid(), tx);
        }
    }

    let mempool_txid = client.get_raw_mempool()?;
    info!("asking mempool txs {:?}", mempool_txid.len());
    let mut count = 0;
    let mut mempool_bucket = MempoolBuckets::new(50, 500.0);
    for txid in mempool_txid.iter() {
        // get_raw_transaction should always work for mempool txs, however some tx may have been replaced
        if let Ok(tx) = client.get_raw_transaction(txid, None) {
            let prev_out_value: Vec<_> = tx
                .input
                .iter()
                .filter_map(|i| {
                    txs.get(&i.previous_output.txid)
                        .map(|tx| tx.output[i.previous_output.vout as usize].value)
                })
                .collect();
            if prev_out_value.len() == tx.input.len() {
                count += 1;
                let sum_input: u64 = prev_out_value.iter().sum();
                let sum_outut: u64 = tx.output.iter().map(|o| o.value).sum();
                let fee = sum_input - sum_outut;
                let fee_rate = (fee as f64) / (tx.get_weight() as f64 / 4.0);
                mempool_bucket.add(*txid, fee_rate);
            }
        }
    }
    info!(
        "mempool_txid:{} of which with input in blocks:{}",
        mempool_txid.len(),
        count
    );

    info!("{:?}", mempool_bucket.buckets);

    for (i, v) in mempool_bucket.buckets.iter().enumerate() {
        data.floats.insert(format!("a{}", i), *v as f32);
    }

    Ok(())
}

fn call_model(inputs: ModelInputs) -> Result<(), Box<dyn Error>> {
    let export_dir = "20210111-164919-model/";

    const MODEL_TAG: &str = "serve";
    let mut graph = Graph::new();
    info!("loading");
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, export_dir)?;
    info!("loaded");

    let sig = bundle
        .meta_graph_def()
        .get_signature(tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    //info!("{:#?}", sig.inputs());
    //info!("{:#?}", sig.outputs());

    let mut float_inputs_args = vec![];
    for (k, v) in inputs.floats {
        let input_info = sig.get_input(&k)?;
        let input_op = graph.operation_by_name_required(&input_info.name().name)?;
        let input_index = input_info.name().index;
        let input_tensor = Tensor::<f32>::new(&[1, 1]).with_values(&[v])?;
        float_inputs_args.push((input_op, input_index, input_tensor));
    }

    let mut int_inputs_args = vec![];
    for (k, v) in inputs.ints {
        let input_info = sig.get_input(&k)?;
        let input_op = graph.operation_by_name_required(&input_info.name().name)?;
        let input_index = input_info.name().index;
        let input_tensor = Tensor::<i64>::new(&[1, 1]).with_values(&[v])?;
        int_inputs_args.push((input_op, input_index, input_tensor));
    }

    let mut run_args = SessionRunArgs::new();
    for input in float_inputs_args.iter() {
        run_args.add_feed(&input.0, input.1, &input.2);
    }
    for input in int_inputs_args.iter() {
        run_args.add_feed(&input.0, input.1, &input.2);
    }

    let output_info = sig.get_output("dense_2")?;
    let output_op = graph.operation_by_name_required(&output_info.name().name)?;
    let output_index = output_info.name().index;
    let output_fetch = run_args.request_fetch(&output_op, output_index);

    bundle.session.run(&mut run_args)?;
    let output = run_args.fetch::<f32>(output_fetch)?;
    info!("{:?}", output[0]);

    Ok(())
}

pub struct MempoolBuckets {
    /// contain the number of elements for bucket ith
    buckets: Vec<u32>,
    /// contain the fee rate limits for every bucket ith
    buckets_limits: Vec<f64>,
    /// in which bucket the Txid is in
    tx_bucket: HashMap<Txid, usize>,
}

impl MempoolBuckets {
    pub fn new(increment_percent: u32, upper_limit: f64) -> Self {
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
        let buckets = vec![0u32; buckets_limits.len()];

        MempoolBuckets {
            buckets,
            buckets_limits,
            tx_bucket: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.tx_bucket.clear();
        for el in self.buckets.iter_mut() {
            *el = 0;
        }
    }

    pub fn add(&mut self, txid: Txid, rate: f64) {
        if rate > 1.0 && self.tx_bucket.get(&txid).is_none() {
            let index = self
                .buckets_limits
                .iter()
                .position(|e| e > &rate)
                .unwrap_or(self.buckets_limits.len() - 1);
            self.buckets[index] += 1;
            self.tx_bucket.insert(txid, index);
        }
    }

    pub fn remove(&mut self, txid: &Txid) {
        if let Some(index) = self.tx_bucket.remove(txid) {
            self.buckets[index] -= 1;
        }
    }

    pub fn number_of_buckets(&self) -> usize {
        self.buckets.len()
    }

    pub fn buckets_str(&self) -> String {
        self.buckets
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn len(&self) -> usize {
        self.tx_bucket.len()
    }

    pub fn txids_set(&self) -> HashSet<&Txid> {
        HashSet::from_iter(self.tx_bucket.keys())
    }
}

// index confirms_in	fee_rate	a0	a1	a2	a3	a4	a5	a6	a7	a8	a9	a10	a11	a12	a13	a14	a15	day_of_week	hour	prediction
// 1422873	2	122.58	476	299	337	409	342	251	248	369	102	216	845	1626	182	81	9	1	3	17	126.660774
