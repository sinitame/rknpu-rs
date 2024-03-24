pub mod context;
pub mod error;
pub mod flags;
pub mod matmul;
pub mod queries;
pub mod tensors;

#[cfg(test)]
mod test {
    use std::sync::{Arc, OnceLock};

    use anyhow::Result;

    use crate::{
        context::{RknnContext, RknnInputOutputNum, RknnSdkVersion},
        flags::RknnExtendedFlag,
        tensors::{
            attributes::RknnTensorAttribute,
            types::{RknnTensorFormat, RknnTensorQuantFormat, RknnTensorType},
        },
    };

    static CTX: OnceLock<Arc<RknnContext>> = OnceLock::new();

    fn load_ctx() -> Arc<RknnContext> {
        Arc::clone(CTX.get_or_init(|| {
            let model_path = "/home/sinitame/mnist_model_quant.rknn";
            let ctx =
                RknnContext::from_model_path(&model_path, RknnExtendedFlag::RKNN_FLAG_PRIOR_HIGH)
                    .unwrap();
            Arc::new(ctx)
        }))
    }

    #[test]
    fn test_sdk_version() -> Result<()> {
        let ctx = load_ctx();
        let sdk_version = ctx.check_version()?;
        let expected_sdk_version = RknnSdkVersion {
            api_version: "1.6.0 (9a7b5d24c@2023-12-13T17:31:11)".to_string(),
            driver_version: "0.8.2".to_string(),
        };
        assert_eq!(sdk_version, expected_sdk_version);
        Ok(())
    }

    #[test]
    fn test_input_attribute() -> Result<()> {
        let ctx = load_ctx();
        let num_input_output = ctx.num_input_outputs()?;
        let expected_num_input_output = RknnInputOutputNum {
            n_input: 1_u32,
            n_output: 1_u32,
        };
        assert_eq!(num_input_output, expected_num_input_output);
        let input_attributes = ctx.input_attribute(0)?;
        let expected_input_attributes = RknnTensorAttribute {
            index: 0,
            dims: vec![1, 28, 28],
            name: "serving_default_input_1:0".to_string(),
            len: 1228800,
            format: RknnTensorFormat::NCHW,
            data_type: RknnTensorType::I8,
            quant_type: RknnTensorQuantFormat::AffineScale(-128, 0.003921569),
            w_stride: 640,
            h_stride: 0,
            size_with_stride: 1228800,
            pass_through: false,
        };
        //TODO: Implement Eq on RknnTensorAttribute
        assert_eq!(input_attributes.index, expected_input_attributes.index);
        assert_eq!(input_attributes.dims, expected_input_attributes.dims);
        assert_eq!(input_attributes.name, expected_input_attributes.name);
        Ok(())
    }

    #[test]
    fn test_output_attribute() -> Result<()> {
        let ctx = load_ctx();
        let num_input_output = ctx.num_input_outputs()?;
        let expected_num_input_output = RknnInputOutputNum {
            n_input: 1_u32,
            n_output: 1_u32,
        };
        assert_eq!(num_input_output, expected_num_input_output);

        let output_1 = RknnTensorAttribute {
            index: 0,
            dims: vec![1, 10],
            name: "StatefulPartitionedCall:0".to_string(),
            len: 10,
            format: RknnTensorFormat::UNDEFINED,
            data_type: RknnTensorType::I8,
            quant_type: RknnTensorQuantFormat::AffineScale(51, 0.17899919),
            w_stride: 0,
            h_stride: 0,
            size_with_stride: 10,
            pass_through: false,
        };
        let expected_outputs = vec![output_1];
        for (out_id, ref_out) in expected_outputs.into_iter().enumerate() {
            let output = ctx.output_attribute(out_id as u32).unwrap();
            assert_eq!(output, ref_out);
        }
        Ok(())
    }
}
