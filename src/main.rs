use bitcoin_fee_model::process_blocks;
use bitcoin_fee_model::{bitcoin, get_model_high, get_model_low};
use bitcoincore_rpc::bitcoin::consensus::serialize;
use bitcoincore_rpc::bitcoin::Block;
use bitcoincore_rpc::bitcoincore_rpc_json::bitcoin::consensus::Decodable;
use bitcoincore_rpc::{Auth, Client, RpcApi};
use log::{debug, info};
use std::convert::TryInto;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;
use structopt::StructOpt;

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
    #[structopt(long, default_value = "https://blockstream.info/api")]
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

    let now = Instant::now();
    let blocks = match opt {
        Command::Node(conf) => get_blocks_from_node(&conf)?,
        Command::Esplora(conf) => get_blocks_from_esplora(&conf)?,
    };
    info!("getting blocks: {:?}", now.elapsed());

    let now = Instant::now();
    let (fee_rates, last_block_time) = process_blocks(&blocks)?;
    info!("process blocks: {:?}", now.elapsed());

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

fn get_blocks_from_esplora(
    conf: &EsploraConfig,
) -> Result<[bitcoin::Block; BLOCKS], Box<dyn Error>> {
    let mut blocks = vec![];

    let mut hash: String = ureq::get(&format!("{}/blocks/tip/hash", conf.url))
        .call()?
        .into_string()?;
    info!("Blockchain hash tip:{}", hash);

    for _ in 0..BLOCKS {
        let block = match check_cache(&hash) {
            Ok(block) => {
                debug!("Cache hit block {}, size: {}", hash, block.get_size());
                block
            }
            Err(_) => {
                let reader = ureq::get(&format!("{}/block/{}/raw", conf.url, hash))
                    .call()?
                    .into_reader();
                let mut buf_reader = BufReader::new(reader);
                let block: Block = Decodable::consensus_decode(&mut buf_reader)?;
                write_cache(&hash, &block)?;
                info!(
                    "Got block {} from esplora, size: {}",
                    hash,
                    block.get_size()
                );
                block
            }
        };
        hash = block.header.prev_blockhash.to_string();

        blocks.push(block);
    }
    blocks.reverse();

    let blocks: [bitcoin::Block; BLOCKS] = blocks.try_into().unwrap();

    Ok(blocks)
}

fn get_blocks_from_node(options: &NodeConfig) -> Result<[bitcoin::Block; BLOCKS], Box<dyn Error>> {
    let client = options.make_rpc_client()?;

    let mut hash = client.get_best_block_hash()?;
    let mut blocks = vec![];

    for _ in 0..BLOCKS {
        let block = match check_cache(&hash.to_string()) {
            Ok(block) => {
                debug!("Cache hit block {}, size: {}", hash, block.get_size());
                block
            }
            Err(_) => {
                let block = client.get_block(&hash)?;
                write_cache(&hash.to_string(), &block)?;
                debug!("Got block {} from node, size: {}", hash, block.get_size());
                block
            }
        };

        hash = block.header.prev_blockhash;
        blocks.push(block);
    }
    blocks.reverse();
    let blocks: [bitcoin::Block; BLOCKS] = blocks.try_into().unwrap();
    info!("Blocks asked to node");

    Ok(blocks)
}

fn check_cache(hash: &str) -> Result<Block, Box<dyn Error>> {
    let mut temp_dir = env::temp_dir();
    temp_dir.push(hash);
    let file = File::open(temp_dir)?;
    let mut buf_reader = BufReader::new(file);
    let block: Block = Decodable::consensus_decode(&mut buf_reader)?;
    Ok(block)
}

fn write_cache(hash: &str, block: &Block) -> Result<(), Box<dyn Error>> {
    let mut temp_dir = env::temp_dir();
    temp_dir.push(hash);
    let mut file = File::create(temp_dir)?;
    file.write(&serialize(block))?;
    Ok(())
}
