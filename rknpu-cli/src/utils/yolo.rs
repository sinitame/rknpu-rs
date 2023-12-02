use std::path::Path;

use anyhow::{bail, Result};
use ndarray::{Array3, Array4, ArrayView3, ArrayView4, Axis};
use rknpu::tensors::types::RknnTensorQuantFormat;

pub fn quantize_value(x: f32, zp: i32, scale: f32) -> u8 {
    (((x / scale).round() as i32) + zp).clamp(u8::min_value() as i32, u8::max_value() as i32) as u8
}

pub fn dequantize_value(x: u8, zp: i32, scale: f32) -> f32 {
    (x as i32 - zp) as f32 * scale
}

pub fn quantize_data(data: Array4<f32>, quant_fmt: RknnTensorQuantFormat) -> Result<Array4<u8>> {
    match quant_fmt {
        RknnTensorQuantFormat::None => bail!("Quantization expected but quant format is None."),
        RknnTensorQuantFormat::AffineScale(zp, scale) => {
            let mut qdata = Array4::from_elem(data.raw_dim(), 0u8);
            qdata
                .as_slice_mut()
                .unwrap()
                .into_iter()
                .zip(data.as_slice().unwrap().into_iter())
                .for_each(|(qx, x)| *qx = quantize_value(*x, zp, scale));
            Ok(qdata)
        }
        RknnTensorQuantFormat::DynamicFixedPoint(_) => unimplemented!(),
    }
}

pub fn load_image<P: AsRef<Path>>(
    image_path: P,
    quant_fmt: RknnTensorQuantFormat,
) -> Result<Vec<u8>> {
    // Input pre-processing
    let image = image::open(image_path).unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 640, 640, ::image::imageops::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 3, 640, 640), |(_, c, y, x)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    });
    let quant_image = quantize_data(image, quant_fmt)?;
    Ok(quant_image.into_raw_vec())
}

const OBJ_CLASS_NUM: usize = 80;
const PROP_BOX_SIZE: usize = 5 + OBJ_CLASS_NUM;

#[derive(Debug, Clone)]
pub struct YoloDetection {
    /// Top-Left Bounds Coordinate in X-Axis
    pub x: f32,
    // Top-Left Bounds Coordinate in Y-Axis
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub class_index: usize,
    pub confidence: f32,
}

impl YoloDetection {
    fn from_raw(raw: &[f32]) -> Self {
        unimplemented!()
    }
}

pub struct Anchor {
    heigh: usize,
    width: usize,
}

// Each hypothesis should be: [cx, cy, w, h, conf, pred_cls(80)]
// For each grid point there are 3 anchors
// So for each point we have 85 * 3 elements -> 255

// Outputs details
// [1, 255, 80, 80]
// - 1: doesn't matter
// - 255: anchor infos [....]
// - (80, 80): Output grid resolution (it means 6400 hypotheses)

pub fn process_result(
    out_buffer: Vec<u8>,
    img_w: usize,
    img_h: usize,
    anchor: Vec<Anchor>,
    grid_x: usize,
    grid_y: usize,
) -> Result<Vec<YoloDetection>> {
    let output = Array3::from_shape_vec((1, 255, grid_x * grid_y), out_buffer)?;
    // We iterate other each grid point
    output
        .axis_iter(Axis(2))
        .map(|anchors_data| {
            let [anchor1, anchor2, anchor3] = anchors_data.as_slice().to_owned().unwrap() else {
                bail!("Error")
            };
            Ok(())
        })
        .collect::<Result<Vec<_>>>()?;
    unimplemented!()
}

#[cfg(test)]
mod test {
    use ndarray::{Array1, Array3, Axis};

    #[test]
    fn test_raw_output_to_array() {
        // N, C, H * W
        // 1, 3, 2, 2
        let raw_output = vec![11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43];

        let output = Array3::from_shape_vec((1, 3, 2 * 2), raw_output).unwrap();
        //let out = output.permuted_axes((0, 2, 1));
        let mut grid_id = 1;
        for row in output.axis_iter(Axis(2)) {
            //let expected = Array1::from_shape_vec(3, (1..4).map(|it| it + 10* grid_id).collect::<Vec<_>>()).unwrap();
            //assert_eq!(row, expected);
            assert!(row.to_owned().as_slice().is_some());
            grid_id += 1;
        }
    }
}

//
//   int stride0 = 8;
//  int grid_h0 = model_in_h / stride0;
//  int grid_w0 = model_in_w / stride0;
//int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];

//#define NMS_THRESH 0.45
//#define BOX_THRESH 0.25
