use std::time::Instant;

use anyhow::Result;
use rknpu::{
    context::{inputs::RknnInput, RknnContext},
    flags::RknnExtendedFlag,
};

use crate::utils::yolo::load_image;

mod utils;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let model_path = &args[1];
    let image_path = &args[2];

    let loading_start = Instant::now();
    let mut rknn_runtime =
        RknnContext::from_model_path(model_path, RknnExtendedFlag::RKNN_FLAG_PRIOR_HIGH)?;
    let loading_duration = loading_start.elapsed();

    dbg!(&loading_duration);
    let input_attribute = rknn_runtime.input_attribute(0)?;
    dbg!(&input_attribute);

    for out_idx in 0..rknn_runtime.num_input_outputs()?.n_output {
        let out_attribute = rknn_runtime.output_attribute(out_idx)?;
        dbg!(&out_attribute.dims);
    }

    let input_quant_fmt = input_attribute.quant_type;
    let input_raw_data = load_image(image_path, input_quant_fmt)?;
    let input = RknnInput {
        index: 0,
        buffer: input_raw_data,
        pass_through: true,
        dtype: input_attribute.data_type,
        fmt: input_attribute.format,
    };

    let start_set_input = Instant::now();
    rknn_runtime.set_inputs(vec![input])?;
    dbg!(start_set_input.elapsed());

    let start_run = Instant::now();
    rknn_runtime.run()?;
    dbg!(start_run.elapsed());

    let outputs = rknn_runtime.get_outputs()?;
    dbg!(outputs.len());
    dbg!(&outputs[0].buffer.len());
    Ok(())
}
