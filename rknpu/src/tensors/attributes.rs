use std::{ffi::CStr, slice};

use anyhow::Result;
use rknpu_sys::_rknn_tensor_attr;

use crate::queries::{QueryObject, RknnQuery};

use super::types::{RknnTensorFormat, RknnTensorQuantFormat, RknnTensorType};

#[derive(Debug, PartialEq, PartialOrd)]
pub struct RknnTensorAttribute {
    // Index of the input/output tensor in the model.
    pub index: usize,
    // Dimensions of the tensor.
    pub dims: Vec<u32>,
    // Name of the tensor.
    pub name: String,
    // Number of elements if the tensor.
    pub len: usize,
    // Format of the tensor.
    pub format: RknnTensorFormat,
    // Data type of the tensor.
    pub data_type: RknnTensorType,
    // Quantization format of the tensor.
    pub quant_type: RknnTensorQuantFormat,
    // stride of the tensor along the width dimension,
    // 0 means equal to width.
    pub w_stride: u32,
    // stride of the tensor along the height dimension,
    // 0 means equal to height.
    pub h_stride: u32,
    // Bytes size of the tensor with stride.
    pub size_with_stride: u32,
    // Pass through mode for rknn_set_io_mem interface
    // - true: data is passed directly to the input node of the model
    // - false: data is converted to an input consistent with the model
    pub pass_through: bool,
}

impl TryFrom<_rknn_tensor_attr> for RknnTensorAttribute {
    type Error = anyhow::Error;
    fn try_from(value: _rknn_tensor_attr) -> Result<Self, Self::Error> {
        let dims =
            unsafe { slice::from_raw_parts(value.dims.as_ptr(), value.n_dims as usize).to_vec() };
        let name = unsafe { CStr::from_ptr(value.name.as_ptr()) };
        let format: RknnTensorFormat = unsafe { std::mem::transmute(value.fmt) };
        let data_type: RknnTensorType = unsafe { std::mem::transmute(value.type_) };
        let quant_type =
            RknnTensorQuantFormat::from_spec(value.qnt_type, value.fl, value.zp, value.scale)?;
        Ok(RknnTensorAttribute {
            index: value.index as usize,
            dims,
            name: name.to_str()?.to_string(),
            len: value.n_elems as usize,
            format,
            data_type,
            quant_type,
            w_stride: value.w_stride,
            h_stride: value.h_stride,
            size_with_stride: value.size_with_stride,
            pass_through: value.pass_through > 0,
        })
    }
}

pub struct RknnInputTensorAttribute(pub RknnTensorAttribute);

impl QueryObject for RknnInputTensorAttribute {
    type RknnPrimitiveType = _rknn_tensor_attr;

    fn primitive_init_value() -> Self::RknnPrimitiveType {
        unsafe { std::mem::zeroed::<_rknn_tensor_attr>() }
    }
    fn from_primitive_type(value: Self::RknnPrimitiveType) -> Result<Self>
    where
        Self: Sized,
    {
        let inner: RknnTensorAttribute = value.try_into()?;
        Ok(Self(inner))
    }

    fn query_flag() -> RknnQuery {
        RknnQuery::RKNN_QUERY_INPUT_ATTR
    }
}

pub struct RknnOutputTensorAttribute(pub RknnTensorAttribute);

impl QueryObject for RknnOutputTensorAttribute {
    type RknnPrimitiveType = _rknn_tensor_attr;

    fn primitive_init_value() -> Self::RknnPrimitiveType {
        unsafe { std::mem::zeroed::<_rknn_tensor_attr>() }
    }
    fn from_primitive_type(value: Self::RknnPrimitiveType) -> Result<Self>
    where
        Self: Sized,
    {
        let inner: RknnTensorAttribute = value.try_into()?;
        Ok(Self(inner))
    }

    fn query_flag() -> RknnQuery {
        RknnQuery::RKNN_QUERY_OUTPUT_ATTR
    }
}
