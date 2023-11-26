mod error;
mod flags;
mod queries;
mod tensor;

use std::{
    ffi::{c_void, CStr},
    io::Read,
    path::Path,
    sync::Arc,
};

use anyhow::{Context, Result};

use error::check_result;
use flags::RknnExtendedFlag;
use queries::RknnQuery;
use rknpu_sys::{
    _rknn_init_extend, _rknn_input_output_num, _rknn_sdk_version, rknn_context, rknn_destroy,
    rknn_init, rknn_query,
};
use tensor::{RknnInputTensorAttribute, RknnOutputTensorAttribute, RknnTensorAttribute};

#[derive(Debug, Eq, PartialEq)]
pub struct RknnSdkVersion {
    pub api_version: String,
    pub driver_version: String,
}

impl QueryObject for RknnSdkVersion {
    type RknnPrimitiveType = _rknn_sdk_version;

    fn primitive_init_value() -> Self::RknnPrimitiveType {
        unsafe { std::mem::zeroed::<_rknn_sdk_version>() }
    }

    fn query_flag() -> RknnQuery {
        RknnQuery::RKNN_QUERY_SDK_VERSION
    }

    fn from_primitive_type(value: Self::RknnPrimitiveType) -> Result<Self>
    where
        Self: Sized,
    {
        let api_version_ctr = unsafe { CStr::from_ptr(value.api_version.as_ptr()) };
        let drv_version_ctr = unsafe { CStr::from_ptr(value.drv_version.as_ptr()) };
        Ok(Self {
            api_version: api_version_ctr.to_str()?.to_string(),
            driver_version: drv_version_ctr.to_str()?.to_string(),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct RknnInputOutputNum {
    pub n_input: u32,
    pub n_output: u32,
}

impl QueryObject for RknnInputOutputNum {
    type RknnPrimitiveType = _rknn_input_output_num;

    fn primitive_init_value() -> Self::RknnPrimitiveType {
        unsafe { std::mem::zeroed::<_rknn_input_output_num>() }
    }
    fn query_flag() -> RknnQuery {
        RknnQuery::RKNN_QUERY_IN_OUT_NUM
    }

    fn from_primitive_type(value: Self::RknnPrimitiveType) -> Result<Self>
    where
        Self: Sized,
    {
        let n_input = value.n_input;
        let n_output = value.n_output;
        Ok(Self { n_input, n_output })
    }
}

pub trait QueryObject {
    type RknnPrimitiveType;
    fn primitive_init_value() -> Self::RknnPrimitiveType;
    fn query_flag() -> RknnQuery;
    fn from_primitive_type(value: Self::RknnPrimitiveType) -> Result<Self>
    where
        Self: Sized;
}

pub struct RknnContext {
    raw: rknn_context,
}

impl RknnContext {
    pub fn from_raw(ctx_ptr: rknn_context) -> Arc<Self> {
        Arc::new(Self { raw: ctx_ptr })
    }

    pub fn query_context<Q: QueryObject>(&self) -> Result<Q> {
        let mut raw_query_result = Q::primitive_init_value();
        let ret = unsafe {
            rknn_query(
                self.raw,
                Q::query_flag() as u32,
                &mut raw_query_result as *mut Q::RknnPrimitiveType as *mut c_void,
                std::mem::size_of::<Q::RknnPrimitiveType>() as u32,
            )
        };
        check_result(ret)?;
        let query_result = Q::from_primitive_type(raw_query_result)?;
        Ok(query_result)
    }
    pub fn check_version(&self) -> Result<RknnSdkVersion> {
        self.query_context::<RknnSdkVersion>()
    }
}

impl Drop for RknnContext {
    fn drop(&mut self) {
        let ret = unsafe { rknn_destroy(self.raw) };
        check_result(ret).unwrap();
    }
}

pub struct Model {
    pub context: Arc<RknnContext>,
}

impl Model {
    pub fn from_path<P: AsRef<Path>>(model_path: P, flag: RknnExtendedFlag) -> Result<Self> {
        let model_data = load_model_data(&model_path)?;
        let size = model_data.len() as u32;
        let mut ctx_ptr = unsafe { std::mem::zeroed::<rknn_context>() };
        let ret = unsafe {
            rknn_init(
                &mut ctx_ptr as *mut rknn_context,
                model_data.as_ptr() as *mut c_void,
                size,
                flag as u32,
                std::ptr::null::<_rknn_init_extend>() as *mut _rknn_init_extend,
            )
        };
        check_result(ret)?;
        let context = RknnContext::from_raw(ctx_ptr);
        Ok(Self { context })
    }

    pub fn num_input_outputs(&self) -> Result<RknnInputOutputNum> {
        self.context.query_context::<RknnInputOutputNum>()
    }
    pub fn input_attribute<const T: usize>(&self) -> Result<RknnTensorAttribute> {
        let input_attribute = self
            .context
            .query_context::<RknnInputTensorAttribute<T>>()?;
        Ok(input_attribute.0)
    }
    pub fn output_attribute<const T: usize>(&self) -> Result<RknnTensorAttribute> {
        let output_attribute = self
            .context
            .query_context::<RknnOutputTensorAttribute<T>>()?;
        Ok(output_attribute.0)
    }
}

fn load_model_data<P: AsRef<Path>>(model_path: P) -> Result<Vec<u8>> {
    let file_metadata =
        std::fs::metadata(&model_path).context("Unable to read model file metadata")?;
    let mut model_file = std::fs::File::open(&model_path)?;
    let mut data = vec![0; file_metadata.len() as usize];
    model_file.read(&mut data)?;
    Ok(data)
}

#[cfg(test)]
mod test {
    use anyhow::Result;

    use crate::{
        flags::RknnExtendedFlag,
        tensor::{RknnTensorAttribute, RknnTensorFormat, RknnTensorQuantFormat, RknnTensorType},
        Model, RknnInputOutputNum, RknnSdkVersion,
    };

    #[test]
    fn test_sdk_version() -> Result<()> {
        // TODO: remove hardcoded path by downloading assets
        // - Don't want to re-download the git repository (but cannot access OUT_DIR in all cases)
        // - Might use the build.rs to copy these files in a cache dir (but not sure it's a good practice)
        let model_path = "/home/sinitame/rknpu2/examples/rknn_yolov5_demo/install/rknn_yolov5_demo_Linux/model/RK3588/yolov5s-640-640.rknn";
        let model = Model::from_path(&model_path, RknnExtendedFlag::RKNN_FLAG_PRIOR_HIGH)?;
        let ctx = model.context;
        let sdk_version = ctx.check_version()?;
        let expected_sdk_version = RknnSdkVersion {
            api_version: "1.5.2 (c6b7b351a@2023-08-23T15:28:22)".to_string(),
            driver_version: "0.8.2".to_string(),
        };
        assert_eq!(sdk_version, expected_sdk_version);
        Ok(())
    }

    #[test]
    fn test_input_attribute() -> Result<()> {
        // TODO: remove hardcoded path by downloading assets
        // - Don't want to re-download the git repository (but cannot access OUT_DIR in all cases)
        // - Might use the build.rs to copy these files in a cache dir (but not sure it's a good practice)
        let model_path = "/home/sinitame/rknpu2/examples/rknn_yolov5_demo/install/rknn_yolov5_demo_Linux/model/RK3588/yolov5s-640-640.rknn";
        let model = Model::from_path(&model_path, RknnExtendedFlag::RKNN_FLAG_PRIOR_HIGH)?;
        let num_input_output = model.num_input_outputs()?;
        let expected_num_input_output = RknnInputOutputNum {
            n_input: 1_u32,
            n_output: 3_u32,
        };
        assert_eq!(num_input_output, expected_num_input_output);
        let input_attributes = model.input_attribute::<0>()?;
        let expected_input_attributes = RknnTensorAttribute {
            index: 0,
            dims: vec![1, 640, 640, 3],
            name: "images".to_string(),
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
        // TODO: remove hardcoded path by downloading assets
        // - Don't want to re-download the git repository (but cannot access OUT_DIR in all cases)
        // - Might use the build.rs to copy these files in a cache dir (but not sure it's a good practice)
        let model_path = "/home/sinitame/rknpu2/examples/rknn_yolov5_demo/install/rknn_yolov5_demo_Linux/model/RK3588/yolov5s-640-640.rknn";
        let model = Model::from_path(&model_path, RknnExtendedFlag::RKNN_FLAG_PRIOR_HIGH)?;
        let num_input_output = model.num_input_outputs()?;
        let expected_num_input_output = RknnInputOutputNum {
            n_input: 1_u32,
            n_output: 3_u32,
        };
        assert_eq!(num_input_output, expected_num_input_output);

        let output_0 = model.output_attribute::<0>().unwrap();
        assert_eq!(output_0.index, 0);
        assert_eq!(output_0.dims, vec![1, 255, 80, 80]);

        let output_1 = model.output_attribute::<1>().unwrap();
        assert_eq!(output_1.index, 1);
        assert_eq!(output_1.dims, vec![1, 255, 40, 40]);

        let output_2 = model.output_attribute::<2>().unwrap();
        assert_eq!(output_2.index, 2);
        assert_eq!(output_2.dims, vec![1, 255, 20, 20]);
        Ok(())
    }
}
