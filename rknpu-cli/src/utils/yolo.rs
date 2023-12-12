use std::path::Path;

use anyhow::{bail, ensure, Result};
use image::Rgb;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::{Array3, Array4, Axis};
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
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    // Apply adjustment is not systematic as it can be applied in the model directly
    fn from_raw_u8(
        raw: &[u8],
        anchor: Option<&Anchor>,
        img_w: f32,
        img_h: f32,
        zp: i32,
        scale: f32,
        apply_adjustment: bool,
    ) -> Result<Self> {
        let ([cx, cy, w, h, conf], classes_confidences) = raw.split_at(5) else {
            bail!("Unexpected number of elements in detection output")
        };
        let box_x = dequantize_value(*cx, zp, scale);
        let box_x = if apply_adjustment {
            Self::grid_sensitivity_adjustment_pos(box_x)
        } else {
            box_x
        };
        let box_y = dequantize_value(*cy, zp, scale);
        let box_y = if apply_adjustment {
            Self::grid_sensitivity_adjustment_pos(box_y)
        } else {
            box_y
        };
        let box_w = dequantize_value(*w, zp, scale);
        let box_w = if let Some(anchor) = anchor {
            Self::grid_sensitivity_adjustment_size(box_w, anchor.width)
        } else {
            box_w
        };
        let box_h = dequantize_value(*h, zp, scale);
        let box_h = if let Some(anchor) = anchor {
            Self::grid_sensitivity_adjustment_size(box_h, anchor.height)
        } else {
            box_h
        };
        Ok(YoloDetection {
            x: (box_x - box_w / 2.0) / img_w,
            y: (box_y - box_h / 2.0) / img_h,
            width: (box_w / img_w),
            height: (box_h / img_h),
            class_index: classes_confidences
                .iter()
                .enumerate()
                .max()
                .map(|(idx, _)| idx)
                .unwrap(),
            confidence: dequantize_value(*conf, zp, scale),
        })
    }

    // Outputs from the predictions head in the original formulation may lead to grid sensitivity
    // issue (output from sigmoid is between 0 and 1 but very low values are required to get 0, and
    // very high values are required to get 1.
    // To mitigate the issue, the following term is used: x * a - (a - 1) * 0.5 but in practice a = 2
    // In the end, output is rescaled from [0, 1] to [-0.5, 1.5] (centered around 0.5)
    fn grid_sensitivity_adjustment_pos(x: f32) -> f32 {
        let alpha = 2.0;
        x * alpha + (alpha - 1.0) * 0.5
    }

    fn grid_sensitivity_adjustment_size(x: f32, anchor_x: usize) -> f32 {
        (x * x * 4.0) * anchor_x as f32
    }
}

pub struct Anchor {
    pub height: usize,
    pub width: usize,
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
    anchors: Option<Vec<Anchor>>,
    grid_x: usize,
    grid_y: usize,
    zp: i32,
    scale: f32,
) -> Result<Vec<YoloDetection>> {
    let n_anchors = anchors.as_ref().map_or_else(
        || Ok(1),
        |it| {
            let n_anchors = it.len();
            ensure!(
                n_anchors == 3,
                "Expected  number of anchors to be equal to 3."
            );
            Ok(n_anchors)
        },
    )?;
    let output =
        Array3::from_shape_vec((1, PROP_BOX_SIZE * n_anchors, grid_x * grid_y), out_buffer)?;
    // We iterate other each grid point
    Ok(output
        .axis_iter(Axis(2))
        .flat_map(|anchors_data| -> Vec<_> {
            anchors_data
                .to_owned()
                .as_slice()
                .unwrap()
                .chunks(PROP_BOX_SIZE)
                .enumerate()
                .map(|(idx, it)| {
                    YoloDetection::from_raw_u8(
                        it,
                        anchors.as_ref().map(|it| &it[idx]),
                        img_w as f32,
                        img_h as f32,
                        zp,
                        scale,
                        true,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>>>()?)
}

pub fn render_detections(
    image_path: &str,
    detections: &Vec<YoloDetection>,
    output_path: &str,
) -> Result<()> {
    let image = image::open(image_path).unwrap();
    let mut image = image.to_rgb8();
    for detection in detections.iter() {
        let x = (detection.x) * image.width() as f32;
        let y = (detection.y) * image.height() as f32;
        let width = (detection.width) * image.width() as f32;
        let height = (detection.height) * image.height() as f32;
        dbg!(&detection.class_index);
        draw_hollow_rect_mut(
            &mut image,
            Rect::at(x as i32, y as i32).of_size(width as u32, height as u32),
            Rgb([255u8, 0u8, 0u8]),
        );
    }

    image.save(output_path).unwrap();

    Ok(())
}

/// Calculate Intersection Over Union (IOU) between two bounding boxes.
pub fn iou(a: &YoloDetection, b: &YoloDetection) -> f32 {
    let area_a = a.area();
    let area_b = b.area();

    let top_left = (a.x.max(b.x), a.y.max(b.y));
    let bottom_right = (a.x + a.width.min(b.width), a.y + a.height.min(b.height));

    let intersection =
        (bottom_right.0 - top_left.0).max(0.0) * (bottom_right.1 - top_left.1).max(0.0);

    intersection / (area_a + area_b - intersection)
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
