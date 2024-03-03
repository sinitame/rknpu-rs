use std::{fs::File, io::Read, time::Instant};

use anyhow::Result;
use rknpu::{
    context::{inputs::RknnInput, RknnContext},
    flags::RknnExtendedFlag,
    tensors::types::{RknnTensorFormat, RknnTensorQuantFormat, RknnTensorType},
};

use crate::utils::yolo::{
    iou, load_image, process_result, render_detections, Anchor, YoloDetection,
};

mod utils;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let model_path = &args[1];
    let image_path = &args[2];
    let min_confidence = args[3].parse::<f32>()?;
    let max_iou = args[4].parse::<f32>()?;

    let loading_start = Instant::now();
    let mut rknn_runtime =
        RknnContext::from_model_path(model_path, RknnExtendedFlag::RKNN_FLAG_PRIOR_HIGH)?;
    let loading_duration = loading_start.elapsed();

    dbg!(&loading_duration);
    let input_attribute = rknn_runtime.input_attribute(0)?;
    dbg!(&input_attribute);

    for out_idx in 0..rknn_runtime.num_input_outputs()?.n_output {
        let out_attribute = rknn_runtime.output_attribute(out_idx)?;
        dbg!(&out_attribute);
    }

    let input_quant_fmt = input_attribute.quant_type;
    let input_raw_data = load_image(image_path, input_quant_fmt)?;
    // Compare input
    let mut file = File::open("/home/sinitame/dev/rknn-toolkit2/rknpu2/examples/rknn_yolov5_demo/install/rknn_yolov5_demo_Linux/input.bin")?;

    let mut file_contents = Vec::new();
    file.read_to_end(&mut file_contents)?;

    let mut diff_indices = vec![];
    let mut diff_values = vec![];

    for (idx, (rs_v, cpp_v)) in input_raw_data.iter().zip(file_contents.iter()).enumerate() {
        let diff = rs_v.abs_diff(*cpp_v);
        if diff > 0 {
            diff_indices.push(idx);
            diff_values.push(diff)
        }
    }
    println!("Num incorrect elements: {}", diff_values.len());
    println!(
        "Proportion of incorrect elements: {}",
        diff_values.len() as f32 / input_raw_data.len() as f32
    );
    println!("Num elements rs: {}", input_raw_data.len());
    println!("Num elements cpp: {}", file_contents.len());

    let input = RknnInput {
        index: 0,
        buffer: file_contents.clone(),
        pass_through: false,
        dtype: RknnTensorType::U8,
        fmt: RknnTensorFormat::NHWC,
    };

    let start_set_input = Instant::now();
    rknn_runtime.set_inputs(vec![input])?;
    dbg!(start_set_input.elapsed());

    let start_run = Instant::now();
    rknn_runtime.run()?;
    dbg!(start_run.elapsed());

    let outputs = rknn_runtime.get_outputs()?;
    let img_w = input_attribute.dims[1];
    let img_h = input_attribute.dims[2];

    // 10, 13, 16, 30, 33, 23
    let anchors0 = vec![
        Anchor {
            width: 10,
            height: 13,
        },
        Anchor {
            width: 16,
            height: 30,
        },
        Anchor {
            width: 33,
            height: 23,
        },
    ];
    // 30, 61, 62, 45, 59, 119
    let anchors1 = vec![
        Anchor {
            width: 30,
            height: 61,
        },
        Anchor {
            width: 62,
            height: 45,
        },
        Anchor {
            width: 59,
            height: 119,
        },
    ];
    // 116, 90, 156, 198, 373, 326
    let anchors2 = vec![
        Anchor {
            width: 116,
            height: 90,
        },
        Anchor {
            width: 156,
            height: 198,
        },
        Anchor {
            width: 373,
            height: 326,
        },
    ];

    let outputs_anchors = vec![(anchors0, 80_usize), (anchors1, 40), (anchors2, 20)];
    let detections = outputs
        .into_iter()
        .zip(outputs_anchors.into_iter())
        .enumerate()
        .flat_map(|(idx, (out, (anchors, grid)))| {
            let (zp, scale) = rknn_runtime
                .output_attribute(idx as u32)
                .unwrap()
                .quant_type
                .zp_scale()
                .unwrap();
            process_result(
                out.buffer,
                img_w as usize,
                img_h as usize,
                Some(anchors),
                grid,
                grid,
                zp,
                scale,
            )
            .unwrap()
        })
        .collect::<Vec<YoloDetection>>();

    let most_probable_detections = detections
        .iter()
        .filter(|it| it.confidence > min_confidence)
        .cloned()
        .collect::<Vec<_>>();
    println!(
        "Number of detection after min_confidence filtering: {}",
        most_probable_detections.len()
    );
    let non_duplicated_detections = most_probable_detections
        .iter()
        .enumerate()
        .filter(|(idx, x)| {
            most_probable_detections
                .iter()
                .skip(*idx + 1)
                .all(|y| iou(x, y) < max_iou)
        })
        .map(|(_, it)| it)
        .cloned()
        .collect::<Vec<_>>();
    println!(
        "Number of detections adter max_iou filtering: {}",
        &non_duplicated_detections.len()
    );
    render_detections(&image_path, &non_duplicated_detections, "out.jpg")?;
    println!("Loading: {loading_duration:?}");
    Ok(())
}
