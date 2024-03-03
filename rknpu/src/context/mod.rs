use std::{
    ffi::{c_void, CStr},
    io::Read,
    path::Path,
};

use anyhow::{Context, Result};
use rknpu_sys::{
    _rknn_init_extend, _rknn_input, _rknn_input_output_num, _rknn_output, _rknn_output_extend,
    _rknn_sdk_version, rknn_context, rknn_destroy, rknn_init, rknn_inputs_set, rknn_outputs_get,
    rknn_query, rknn_run, rknn_run_extend,
};

use crate::{
    error::check_result,
    flags::RknnExtendedFlag,
    queries::{QueryObject, RknnQuery},
    tensors::attributes::{
        RknnInputTensorAttribute, RknnOutputTensorAttribute, RknnTensorAttribute,
    },
};

use self::{inputs::RknnInput, outputs::RknnOuput};

pub mod inputs;
pub mod memory;
pub mod outputs;

pub struct RknnContext {
    raw: rknn_context,
}

impl RknnContext {
    pub fn from_model_path<P: AsRef<Path>>(model_path: P, flag: RknnExtendedFlag) -> Result<Self> {
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
        Ok(Self { raw: ctx_ptr })
    }

    pub fn query_context<Q: QueryObject>(
        &self,
        default: Option<Q::RknnPrimitiveType>,
    ) -> Result<Q> {
        let mut raw_query_result = default.unwrap_or(Q::primitive_init_value());
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
        self.query_context::<RknnSdkVersion>(None)
    }

    pub fn num_input_outputs(&self) -> Result<RknnInputOutputNum> {
        self.query_context::<RknnInputOutputNum>(None)
    }

    pub fn input_attribute(&self, index: u32) -> Result<RknnTensorAttribute> {
        let mut default = RknnInputTensorAttribute::primitive_init_value();
        default.index = index;
        let input_attribute = self.query_context::<RknnInputTensorAttribute>(Some(default))?;
        Ok(input_attribute.0)
    }

    pub fn run(&mut self) -> Result<()> {
        let ret = unsafe {
            rknn_run(
                self.raw,
                std::ptr::null::<rknn_run_extend>() as *mut rknn_run_extend,
            )
        };
        check_result(ret)?;
        Ok(())
    }

    pub fn set_inputs(&mut self, inputs: Vec<RknnInput>) -> Result<()> {
        let raw_inputs: Vec<_rknn_input> = inputs.into_iter().map(|it| it.into()).collect();
        let ret = unsafe {
            rknn_inputs_set(
                self.raw,
                raw_inputs.len() as u32,
                raw_inputs.as_ptr() as *mut _rknn_input,
            )
        };
        check_result(ret)?;
        Ok(())
    }

    pub fn get_outputs(&mut self) -> Result<Vec<RknnOuput>> {
        let n_outputs = self.num_input_outputs()?.n_output;
        let mut raw_outputs =
            vec![unsafe { std::mem::zeroed::<_rknn_output>() }; n_outputs as usize];
        //let mut raw_outputs: Vec<_rknn_output> = Vec::with_capacity(n_outputs as usize);
        raw_outputs.iter_mut().enumerate().for_each(|(idx, it)| {
            it.index = idx as u32;
            it.want_float = false as u8;
        });
        let ret = unsafe {
            rknn_outputs_get(
                self.raw,
                n_outputs,
                raw_outputs.as_ptr() as *mut _rknn_output,
                std::ptr::null::<_rknn_output_extend>() as *mut _rknn_output_extend,
            )
        };
        check_result(ret)?;
        let outputs = raw_outputs.into_iter().map(|it| it.into()).collect();
        Ok(outputs)
    }
    pub fn output_attribute(&self, index: u32) -> Result<RknnTensorAttribute> {
        let mut default = RknnOutputTensorAttribute::primitive_init_value();
        default.index = index;
        let output_attribute = self.query_context::<RknnOutputTensorAttribute>(Some(default))?;
        Ok(output_attribute.0)
    }
}

impl Drop for RknnContext {
    fn drop(&mut self) {
        let ret = unsafe { rknn_destroy(self.raw) };
        check_result(ret).unwrap();
    }
}

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

fn load_model_data<P: AsRef<Path>>(model_path: P) -> Result<Vec<u8>> {
    let file_metadata =
        std::fs::metadata(&model_path).context("Unable to read model file metadata")?;
    let mut model_file = std::fs::File::open(&model_path)?;
    let mut data = vec![0; file_metadata.len() as usize];
    model_file.read(&mut data)?;
    Ok(data)
}
