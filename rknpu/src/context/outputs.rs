use std::ffi::c_void;

use rknpu_sys::_rknn_output;

#[derive(Debug)]
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
    pub buffer: Vec<u8>,
}

impl From<_rknn_output> for RknnOuput {
    fn from(value: _rknn_output) -> Self {
        let want_float = if value.want_float > 0 { true } else { false };
        let is_prealloc = if value.is_prealloc > 0 { true } else { false };
        Self {
            want_float,
            is_prealloc,
            index: value.index,
            buffer: unsafe {
                std::slice::from_raw_parts(value.buf as *mut u8, value.size as usize).to_vec()
            },
        }
    }
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
