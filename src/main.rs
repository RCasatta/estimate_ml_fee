mod blocks;
mod transactions;

use crate::blocks::BlocksBuckets;
use bitcoincore_rpc::{Auth, Client, RpcApi};
use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::ops::Sub;
use std::path::PathBuf;
use structopt::StructOpt;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

/*
Next: 129.6 sat/byte  $12.47
1h:   122.5 sat/byte  $11.80
6h:   50.6 sat/byte  $4.88
12h:  21.1 sat/byte  $2.03
1d:   14.1 sat/byte  $1.36
3d:    8.0 sat/byte  $0.77
1wk:   1.1 sat/byte  $0.11
Min:   1.9 sat/byte  $0.19
Block height: 666,154
*/

const BLOCK_TARGETS: [u16; 8] = [1, 3, 6, 36, 72, 144, 432, 1008];

#[derive(Serialize, Deserialize, Debug)]
pub struct FieldsDescribe {
    mean: HashMap<String, f32>,
    std: HashMap<String, f32>,
    fields: Vec<String>,
}

#[derive(StructOpt, Debug)]
pub struct EstimateOptions {
    /// the number of blocks I am ok to wait for, must be between 1 (included) and 1008 (included)
    #[structopt(long)]
    pub block_target: u16,

    /// The path where the model is saved
    #[structopt(long)]
    pub model_path: PathBuf,

    #[structopt(flatten)]
    pub node_config: NodeConfig,

    /// Stop considering `limit` tx of the mempool, by default consider all txs
    #[structopt(long)]
    pub limit: Option<usize>,
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

    let mut json_path = options.model_path.clone();
    json_path.push("mean-std.json");

    let mut model_path = options.model_path.clone();
    model_path.push("model");

    let mut file = File::open(json_path)?;
    let mut buffer = vec![];
    file.read_to_end(&mut buffer)?;
    let mean_std_fields: FieldsDescribe = serde_json::from_slice(&buffer)?;
    info!("{:?}", mean_std_fields);

    let block_target = options.block_target;

    if block_target == 0 || block_target > 1008 {
        error!("--blocks-target should be between 1 (included) and 1008 (included)");
        return Ok(());
    }

    let model = load_model(&model_path)?;

    let mut inputs_map: HashMap<String, f32> = HashMap::new();
    let utc: DateTime<Utc> = Utc::now();
    //inputs.insert()
    let timestamp = utc.timestamp() as u32;
    let (buckets, last_block_time) = calculate_buckets(&options)?;

    inputs_map.insert(
        "day_of_week".to_string(),
        utc.weekday().num_days_from_monday() as f32,
    );
    inputs_map.insert("hour".to_string(), utc.hour() as f32);
    inputs_map.insert(
        "delta_last".to_string(),
        (timestamp - last_block_time) as f32,
    );
    //
    for i in 0..=15 {
        inputs_map.insert(format!("b{}", i), buckets[i] as f32);
    }
    info!("inputs_map: {:?}", inputs_map);
    let mut block_targets = BLOCK_TARGETS.to_vec();
    block_targets.push(block_target);
    block_targets.sort();

    for block_target in block_targets {
        let inputs = calculate_inputs(block_target, &inputs_map, &mean_std_fields);
        let estimate = predict(&inputs, &model)?;
        info!(
            "Estimated fee to enter in {} blocks is {:?} sat/vbyte",
            block_target, estimate
        );
    }

    Ok(())
}

fn calculate_inputs(
    block_target: u16,
    inputs_map: &HashMap<String, f32>,
    fields: &FieldsDescribe,
) -> Vec<f32> {
    let mut result = vec![];
    let mut inputs_map = inputs_map.clone();
    inputs_map.insert("confirms_in".to_string(), block_target as f32);
    for field in fields.fields.iter() {
        let x = inputs_map.get(field).unwrap();
        let std = fields.std.get(field).unwrap();
        let mean = fields.mean.get(field).unwrap();
        let res = (x - mean) / std;
        debug!("{}:{} norm:{}", field, x, res);
        result.push(res)
    }
    result
}

fn calculate_buckets(options: &EstimateOptions) -> Result<(Vec<u64>, u32), Box<dyn Error>> {
    let old: DateTime<Utc> = Utc::now().sub(Duration::hours(2));
    info!("old is {}", old);
    let client = options.node_config.make_rpc_client()?;

    let mut blocks_to_ask = vec![];
    let best_block_hash = client.get_best_block_hash()?;
    let mut last = best_block_hash;
    for _ in 0..10 {
        blocks_to_ask.push(last);
        let block_info = client.get_block_info(&last)?;
        last = block_info.previousblockhash.expect("found a stale block");
    }

    info!("asking blocks {:?}", blocks_to_ask);
    let mut bb = BlocksBuckets::new(50, 500.0, 10);
    let mut time = None;
    for hash in blocks_to_ask {
        let block = client.get_block(&hash)?;
        if block.txdata.len() > 1 && time.is_none() {
            time = Some(block.header.time);
        }
        bb.add(block);
    }
    let buckets = bb.get_buckets().clone().unwrap();

    Ok((buckets, time.unwrap()))
}

fn load_model(model_path: &PathBuf) -> Result<(Graph, SavedModelBundle), Box<dyn Error>> {
    const MODEL_TAG: &str = "serve";
    let mut graph = Graph::new();
    info!("loading");
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, model_path)?;
    info!("loaded");
    Ok((graph, bundle))
}

fn predict(inputs: &[f32], model: &(Graph, SavedModelBundle)) -> Result<f32, Box<dyn Error>> {
    let (graph, bundle) = model;
    let sig = bundle
        .meta_graph_def()
        .get_signature(tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    //info!("{:#?}", sig.inputs());
    //info!("{:#?}", sig.outputs());

    let input_info = sig.get_input("dense_input")?;
    let input_op = graph.operation_by_name_required(&input_info.name().name)?;
    let input_index = input_info.name().index;
    let input_tensor = Tensor::<f32>::new(&[1, 20]).with_values(inputs)?;
    let mut run_args = SessionRunArgs::new();
    run_args.add_feed(&input_op, input_index, &input_tensor);

    let output_info = sig.get_output("dense_2")?;
    let output_op = graph.operation_by_name_required(&output_info.name().name)?;
    let output_index = output_info.name().index;
    let output_fetch = run_args.request_fetch(&output_op, output_index);

    bundle.session.run(&mut run_args)?;
    let output = run_args.fetch::<f32>(output_fetch)?;

    Ok(output[0])
}
