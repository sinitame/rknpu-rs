use std::sync::Mutex;

use rknpu_sys::_rknn_tensor_memory;

pub struct RknnTensorMemory {
    raw: Mutex<_rknn_tensor_memory>,
}

impl RknnTensorMemory {
    pub fn from_raw(raw: _rknn_tensor_memory) -> Self {
        Self {
            raw: Mutex::new(raw),
        }
    }

    pub fn to_raw(self) -> _rknn_tensor_memory {
        self.raw.into_inner().unwrap()
    }
}

unsafe impl Send for RknnTensorMemory {}
unsafe impl Sync for RknnTensorMemory {}
