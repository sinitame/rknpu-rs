mod error;
mod flags;
mod queries;

use std::{
    ffi::{c_void, CStr},
    io::Read,
    path::Path,
    sync::Arc,
};

use anyhow::{Context, Result};

use error::check_error;
use flags::RknnExtendedFlag;
use queries::RknnQuery;
use rknpu_sys::{
    _rknn_init_extend, _rknn_sdk_version, rknn_context, rknn_destroy, rknn_init, rknn_query,
};

#[derive(Debug)]
pub struct RknnSdkVersion {
    pub api_version: String,
    pub driver_version: String,
}

impl TryFrom<_rknn_sdk_version> for RknnSdkVersion {
    type Error = anyhow::Error;
    fn try_from(value: _rknn_sdk_version) -> Result<Self, Self::Error> {
        let api_version_ctr = unsafe { CStr::from_ptr(value.api_version.as_ptr()) };
        let drv_version_ctr = unsafe { CStr::from_ptr(value.drv_version.as_ptr()) };
        Ok(Self {
            api_version: api_version_ctr.to_str()?.to_string(),
            driver_version: drv_version_ctr.to_str()?.to_string(),
        })
    }
}

pub struct RknnContext {
    raw: rknn_context,
}

impl RknnContext {
    pub fn from_raw(ctx_ptr: rknn_context) -> Arc<Self> {
        Arc::new(Self { raw: ctx_ptr })
    }

    pub fn check_version(&self) -> Result<RknnSdkVersion> {
        let mut raw_sdk_version: _rknn_sdk_version;
        let ret = unsafe {
            rknn_query(
                self.raw,
                RknnQuery::RKNN_QUERY_SDK_VERSION as u32,
                &mut raw_sdk_version as *mut _rknn_sdk_version as *mut c_void,
                std::mem::size_of::<_rknn_sdk_version>() as u32,
            )
        };
        check_error(ret)?;
        let sdk_version = raw_sdk_version.try_into()?;
        Ok(sdk_version)
    }
}

impl Drop for RknnContext {
    fn drop(&mut self) {
        let ret = unsafe { rknn_destroy(self.raw) };
    }
}

pub struct Model {
    context: Arc<RknnContext>,
}

impl Model {
    fn from_path<P: AsRef<Path>>(model_path: P, flag: RknnExtendedFlag) -> Result<Self> {
        let mut model_data = load_model_data(&model_path)?;
        let size = model_data.len() as u32;
        let mut ctx_ptr: rknn_context;
        let ret = unsafe {
            rknn_init(
                &mut ctx_ptr as *mut rknn_context,
                model_data.as_ptr() as *mut c_void,
                size,
                flag as u32,
                std::ptr::null::<_rknn_init_extend>() as *mut _rknn_init_extend,
            )
        };
        check_error(ret)?;
        let context = RknnContext::from_raw(ctx_ptr);
        Ok(Self { context })
    }
}

fn load_model_data<P: AsRef<Path>>(model_path: P) -> Result<&[u8]> {
    let file_metadata =
        std::fs::metadata(&model_path).context("Unable to read model file metadata")?;
    let mut model_file = std::fs::File::open(&model_path)?;
    let mut data = vec![0; file_metadata.len() as usize];
    model_file.read(&mut data)?;
    Ok(data.as_slice())
}
