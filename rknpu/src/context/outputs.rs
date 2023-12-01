use std::ffi::c_void;

use rknpu_sys::_rknn_output;

pub struct RknnOuput {
    /// Transfert output data to float
    want_float: bool,
    /// Indicate if buffer is pre-allocated:
    /// - true: index and buffer needs to be set
    /// - fasle: index and buffer don't needs to be set
    is_prealloc: bool,
    /// Output index
    index: u32,
    /// Output data buffer
    buffer: Vec<u8>,
}

impl From<RknnOuput> for _rknn_output {
    fn from(value: RknnOuput) -> Self {
        Self {
            index: value.index,
            want_float: value.want_float as u8,
            is_prealloc: value.is_prealloc as u8,
            buf: value.buffer.as_ptr() as *mut c_void,
            size: value.buffer.len() as u32,
        }
    }
}
