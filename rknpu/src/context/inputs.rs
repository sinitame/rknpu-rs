use std::ffi::c_void;

use rknpu_sys::_rknn_input;

use crate::tensors::types::{RknnTensorFormat, RknnTensorType};

pub struct RknnInput {
    /// Input index.
    pub index: u32,
    /// Input data buffer.
    pub buffer: Vec<u8>,
    /// Pass through mode
    /// - true: the data buffer is passed directly to the input node of the rknn model without any
    /// conversion. The following variables don't need to be set.
    /// - false: the data buffer is converted into an input consistent with the model according to
    /// the following type and format. The following variables need to be set.
    pub pass_through: bool,
    /// Data type of the input data.
    pub dtype: RknnTensorType,
    /// Data format of the input data (NPU accepts NCHW by default, other format will require a
    /// conversion in the driver).
    pub fmt: RknnTensorFormat,
}

impl From<RknnInput> for _rknn_input {
    fn from(value: RknnInput) -> Self {
        Self {
            index: value.index,
            buf: value.buffer.as_ptr() as *mut c_void,
            size: value.buffer.len() as u32,
            pass_through: value.pass_through as u8,
            type_: value.dtype as u32,
            fmt: value.fmt as u32,
        }
    }
}
