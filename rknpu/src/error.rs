use std::ffi::c_int;

use thiserror::Error;

use rknpu_sys::{
    RKNN_ERR_CTX_INVALID, RKNN_ERR_DEVICE_UNAVAILABLE, RKNN_ERR_DEVICE_UNMATCH, RKNN_ERR_FAIL,
    RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION, RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL,
    RKNN_ERR_INPUT_INVALID, RKNN_ERR_MALLOC_FAIL, RKNN_ERR_MODEL_INVALID, RKNN_ERR_OUTPUT_INVALID,
    RKNN_ERR_PARAM_INVALID, RKNN_ERR_TARGET_PLATFORM_UNMATCH, RKNN_ERR_TIMEOUT,
};

#[derive(Debug, Error)]
#[allow(non_camel_case_types)]
pub enum RknnError {
    #[error("Success")]
    RKNN_SUCC,
    #[error("Execution failed.")]
    RKNN_ERR_FAIL,
    #[error("Execution timeout.")]
    RKNN_ERR_TIMEOUT,
    #[error("Device is unavailable.")]
    RKNN_ERR_DEVICE_UNAVAILABLE,
    #[error("Memory malloc fail.")]
    RKNN_ERR_MALLOC_FAIL,
    #[error("Parameter is invalid.")]
    RKNN_ERR_PARAM_INVALID,
    #[error("Model is invalid.")]
    RKNN_ERR_MODEL_INVALID,
    #[error("Context is invalid.")]
    RKNN_ERR_CTX_INVALID,
    #[error("Input is invalid.")]
    RKNN_ERR_INPUT_INVALID,
    #[error("Output is invalid.")]
    RKNN_ERR_OUTPUT_INVALID,
    #[error("The device is unmatch, please update rknn sdk and npu driver/firmware.")]
    RKNN_ERR_DEVICE_UNMATCH,
    #[error("This RKNN model use pre_compile mode, but not compatible with current driver.")]
    RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL,
    #[error("This RKNN model set optimization level, but not compatible with current driver.")]
    RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION,
    #[error("This RKNN model set target platform, but not compatible with current platform.")]
    RKNN_ERR_TARGET_PLATFORM_UNMATCH,
    #[error("Unknown error.")]
    UNKNOWN,
}

impl From<c_int> for RknnError {
    fn from(value: c_int) -> Self {
        match value {
            0 => RknnError::RKNN_SUCC,
            -1 => RknnError::RKNN_ERR_FAIL,
            -2 => RknnError::RKNN_ERR_TIMEOUT,
            -3 => RknnError::RKNN_ERR_DEVICE_UNAVAILABLE,
            -4 => RknnError::RKNN_ERR_MALLOC_FAIL,
            -5 => RknnError::RKNN_ERR_PARAM_INVALID,
            -6 => RknnError::RKNN_ERR_MODEL_INVALID,
            -7 => RknnError::RKNN_ERR_CTX_INVALID,
            -8 => RknnError::RKNN_ERR_INPUT_INVALID,
            -9 => RknnError::RKNN_ERR_OUTPUT_INVALID,
            -10 => RknnError::RKNN_ERR_DEVICE_UNMATCH,
            -11 => RknnError::RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL,
            -12 => RknnError::RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION,
            -13 => RknnError::RKNN_ERR_TARGET_PLATFORM_UNMATCH,
            _ => RknnError::UNKNOWN,
        }
    }
}

pub(crate) fn check_error(rknn_error: c_int) -> Result<(), RknnError> {
    let error = rknn_error.into();
    match error {
        RknnError::RKNN_SUCC => Ok(()),
        err => Err(err),
    }
}
