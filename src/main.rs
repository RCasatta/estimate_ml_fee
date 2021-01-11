use std::error::Error;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    let export_dir = "saved_model/my_model/";

    const MODEL_TAG: &str = "serve";
    let mut graph = Graph::new();
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, export_dir)?;
    let sig = bundle
        .meta_graph_def()
        .get_signature(tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    println!("{:?}", sig.inputs());
    println!("{:?}", sig.outputs());
    let input_info = sig.get_input("dense_input")?;
    let output_info = sig.get_output("dense_2")?;
    let input_op = graph.operation_by_name_required(&input_info.name().name)?;
    let output_op = graph.operation_by_name_required(&output_info.name().name)?;
    let input_index = input_info.name().index;
    let output_index = output_info.name().index;

    let input_tensor = Tensor::<f32>::new(&[1, 9]).with_values(&[
        -0.869348f32,
        -0.721914,
        -0.679055,
        -0.432815,
        1.090181,
        1.660094,
        -0.465148,
        -0.495225,
        0.774676,
    ])?;
    let mut run_args = SessionRunArgs::new();
    run_args.add_feed(&input_op, input_index, &input_tensor);
    let output_fetch = run_args.request_fetch(&output_op, output_index);
    bundle.session.run(&mut run_args)?;
    let output = run_args.fetch::<f32>(output_fetch)?;
    println!("{:?}", output[0]);

    Ok(())
}
