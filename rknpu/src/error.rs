use std::ffi::c_int;

use thiserror::Error;

use rknpu_sys::{
    RKNN_ERR_CTX_INVALID, RKNN_ERR_DEVICE_UNAVAILABLE, RKNN_ERR_DEVICE_UNMATCH, RKNN_ERR_FAIL,
    RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION, RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL,
    RKNN_ERR_INPUT_INVALID, RKNN_ERR_MALLOC_FAIL, RKNN_ERR_MODEL_INVALID, RKNN_ERR_OUTPUT_INVALID,
    RKNN_ERR_PARAM_INVALID, RKNN_ERR_TARGET_PLATFORM_UNMATCH, RKNN_ERR_TIMEOUT,
};

#[derive(Debug, Error)]
pub enum RknnError {
    #[error("Execution failed.")]
    Fail,
    #[error("Execution timeout.")]
    Timeout,
    #[error("Device is unavailable.")]
    UnavailableDevice,
    #[error("Memory malloc fail.")]
    MallocFail,
    #[error("Parameter is invalid.")]
    InvalidParameter,
    #[error("Model is invalid.")]
    InvalidModel,
    #[error("Context is invalid.")]
    InvalidContext,
    #[error("Input is invalid.")]
    InvalidInput,
    #[error("Output is invalid.")]
    InvalidOutput,
    #[error("The device is unmatch, please update rknn sdk and npu driver/firmware.")]
    UnmatchedDevice,
    #[error("This RKNN model use pre_compile mode, but not compatible with current driver.")]
    IncompatibleModel,
    #[error("This RKNN model set optimization level, but not compatible with current driver.")]
    IncompatibleOptimization,
    #[error("This RKNN model set target platform, but not compatible with current platform.")]
    UnmatchedTargetPlatform,
    #[error("Unknown error.")]
    Unknown,
}

#[allow(non_snake_case)]
impl From<c_int> for RknnError {
    fn from(value: c_int) -> Self {
        match value as i32 {
            RKNN_ERR_FAIL => RknnError::Fail,
            RKNN_ERR_TIMEOUT => RknnError::Timeout,
            RKNN_ERR_DEVICE_UNAVAILABLE => RknnError::UnavailableDevice,
            RKNN_ERR_MALLOC_FAIL => RknnError::MallocFail,
            RKNN_ERR_PARAM_INVALID => RknnError::InvalidParameter,
            RKNN_ERR_MODEL_INVALID => RknnError::InvalidModel,
            RKNN_ERR_CTX_INVALID => RknnError::InvalidContext,
            RKNN_ERR_INPUT_INVALID => RknnError::InvalidInput,
            RKNN_ERR_OUTPUT_INVALID => RknnError::InvalidOutput,
            RKNN_ERR_DEVICE_UNMATCH => RknnError::UnmatchedDevice,
            RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL => RknnError::IncompatibleModel,
            RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION => RknnError::IncompatibleOptimization,
            RKNN_ERR_TARGET_PLATFORM_UNMATCH => RknnError::UnmatchedTargetPlatform,
            _ => RknnError::Unknown,
        }
    }
}

pub(crate) fn check_result(rknn_result: c_int) -> Result<(), RknnError> {
    if rknn_result == 0 {
        return Ok(());
    } else {
        let error = rknn_result.into();
        return Err(error);
    }
}
