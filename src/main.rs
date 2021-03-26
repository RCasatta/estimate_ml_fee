use bitcoin_fee_model::process_blocks;
use bitcoin_fee_model::{bitcoin, get_model_high, get_model_low};
use bitcoincore_rpc::{Auth, Client, RpcApi};
use log::{info, debug};
use std::convert::TryInto;
use std::error::Error;
use std::time::Instant;
use structopt::StructOpt;
use std::io::{Read, BufReader};
use bitcoincore_rpc::bitcoincore_rpc_json::bitcoin::consensus::{deserialize, Decodable};
use bitcoincore_rpc::bitcoin::Block;

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

const BLOCK_TARGETS: [u16; 9] = [1, 2, 3, 6, 36, 72, 144, 432, 1008];

#[derive(StructOpt, Debug)]
enum Command {
    /// Gather last 10 block through a local running node
    Node(NodeConfig),
    /// Gather last 10 block from an esplora service
    Esplora(EsploraConfig),
}

#[derive(StructOpt, Debug)]
pub struct EsploraConfig {
    #[structopt(long, default_value="https://blockstream.info/api/")]
    pub url: String,
}

#[derive(StructOpt, Debug)]
pub struct NodeConfig {
    /// Rpc address eg. "http://127.0.0.1:8332"
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
    let opt = Command::from_args();
    let blocks = match opt {
        Command::Node(conf) => get_blocks_from_node(&conf)?,
        Command::Esplora(conf) => get_blocks_from_esplora(&conf)?,
    };

    let (fee_rates, last_block_time) = process_blocks(&blocks)?;
    let now = Instant::now();

    let model = bitcoin_fee_model::FeeModel::new(get_model_low(), get_model_high());
    info!("load elapsed: {:?}", now.elapsed());
    let now = Instant::now();

    for i in BLOCK_TARGETS.iter() {
        let f = model.estimate(*i, None, &fee_rates, last_block_time)?;
        info!("i:{} f:{}", i, f);
    }

    info!("estimate elapsed: {:?}", now.elapsed());

    Ok(())
}

const BLOCKS: usize = 10;

fn get_blocks_from_esplora(conf: &EsploraConfig) -> Result<[bitcoin::Block; BLOCKS], Box<dyn Error>> {
    let mut blocks = vec![];

    let height: u32 = ureq::get(&format!("{}/blocks/tip/height", conf.url))
        .call()?
        .into_string()?.parse()?;
    debug!("Blockchain height:{}", height);

    for i in height-10..height {
        let hash = ureq::get(&format!("{}/block-height/{}", conf.url, i))
            .call()?
            .into_string()?;
        debug!("Block height {} hash: {}", i, hash);

        let reader = ureq::get(&format!("{}/block/{}/raw", conf.url, hash))
            .call()?.into_reader();
        let mut buf_reader = BufReader::new(reader);
        let block: Block = Decodable::consensus_decode(&mut buf_reader)?;
        debug!("Got block {}, size: {}", hash, block.get_size());
        blocks.push(block);
    }

    let blocks: [bitcoin::Block; BLOCKS] = blocks.try_into().unwrap();

    Ok(blocks)
}

fn get_blocks_from_node(options: &NodeConfig) -> Result<[bitcoin::Block; BLOCKS], Box<dyn Error>> {
    let client = options.make_rpc_client()?;

    let mut hash = client.get_best_block_hash()?;
    let mut blocks = vec![];

    for _ in 0..BLOCKS {
        let block = client.get_block(&hash)?;
        hash = block.header.prev_blockhash;
        blocks.push(block);
    }
    blocks.reverse();
    let blocks: [bitcoin::Block; BLOCKS] = blocks.try_into().unwrap();
    info!("Blocks asked to node");

    Ok(blocks)
}

/*
a1 = input.dot(weights['dense/kernel:0'])
a2 = a1 + weights['dense/bias:0']
a3 = a2.clip(0) # relu

b1 = a3.dot(weights['dense_1/kernel:0'])
b2 = b1 + weights['dense_1/bias:0']
b3 = b2.clip(0) # relu

c1 = b3.dot(weights['dense_2/kernel:0'])
c2 = c1 + weights['dense_2/bias:0']

c2
*/

/*
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
*/
