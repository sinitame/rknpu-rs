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

#[derive(Debug, Eq, PartialEq)]
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
        let mut raw_sdk_version = unsafe { std::mem::zeroed::<_rknn_sdk_version>() };
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
        check_error(ret).unwrap();
    }
}

pub struct Model {
    pub context: Arc<RknnContext>,
}

impl Model {
    fn from_path<P: AsRef<Path>>(model_path: P, flag: RknnExtendedFlag) -> Result<Self> {
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
        check_error(ret)?;
        let context = RknnContext::from_raw(ctx_ptr);
        Ok(Self { context })
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

    use crate::{Model, flags::RknnExtendedFlag, RknnSdkVersion};

    #[test]
    fn test_sdk_version() -> Result<()>{
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
}
