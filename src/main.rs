use std::error::Error;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};
use std::collections::HashMap;
// index confirms_in	fee_rate	a0	a1	a2	a3	a4	a5	a6	a7	a8	a9	a10	a11	a12	a13	a14	a15	day_of_week	hour	prediction
// 1422873	2	122.58	476	299	337	409	342	251	248	369	102	216	845	1626	182	81	9	1	3	17	126.660774

fn main() -> Result<(), Box<dyn Error>> {
    let export_dir = "20210111-164919-model/";

    const MODEL_TAG: &str = "serve";
    let mut graph = Graph::new();
    println!("loading");
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, export_dir)?;
    println!("loaded");
/*
    let mut float_inputs = HashMap::new();
    float_inputs.insert("confirms_in".to_string(), 1.0f32);
    for i in 0..=15 {
        float_inputs.insert(format!("a{}", i), 1.0f32);
    }
    let mut int_inputs = HashMap::new();
    int_inputs.insert("day_of_week".to_string(), 0i64);
    int_inputs.insert("hour".to_string(), 0);
*/
    let mut float_inputs = HashMap::new();
    float_inputs.insert("confirms_in".to_string(), 2.0f32);
    let buckets = "476	299	337	409	342	251	248	369	102	216	845	1626	182	81	9	1";
    for (i,b)  in buckets.split("\t").enumerate() {
        float_inputs.insert(format!("a{}", i), b.parse()?);
    }

    let mut int_inputs = HashMap::new();
    int_inputs.insert("day_of_week".to_string(), 3i64);
    int_inputs.insert("hour".to_string(), 17);

    let sig = bundle
        .meta_graph_def()
        .get_signature(tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    //println!("{:#?}", sig.inputs());
    //println!("{:#?}", sig.outputs());

    let mut float_inputs_args= vec![];
    for (k,v) in float_inputs {
        let input_info = sig.get_input(&k)?;
        let input_op = graph.operation_by_name_required(&input_info.name().name)?;
        let input_index = input_info.name().index;
        let input_tensor = Tensor::<f32>::new(&[1, 1]).with_values(&[v])?;
        float_inputs_args.push((input_op, input_index, input_tensor));
    }

    let mut int_inputs_args= vec![];
    for (k,v) in int_inputs {
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
    println!("{:?}", output[0]);

    Ok(())
}
