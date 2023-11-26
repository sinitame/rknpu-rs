use std::{ffi::CStr, slice};

use anyhow::{bail, Result};
use rknpu_sys::{
    _rknn_tensor_attr, _rknn_tensor_format_RKNN_TENSOR_FORMAT_MAX,
    _rknn_tensor_format_RKNN_TENSOR_NC1HWC2, _rknn_tensor_format_RKNN_TENSOR_NCHW,
    _rknn_tensor_format_RKNN_TENSOR_NHWC, _rknn_tensor_format_RKNN_TENSOR_UNDEFINED,
    _rknn_tensor_qnt_type, _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC,
    _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_DFP, _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_NONE,
    _rknn_tensor_type_RKNN_TENSOR_BOOL, _rknn_tensor_type_RKNN_TENSOR_FLOAT16,
    _rknn_tensor_type_RKNN_TENSOR_FLOAT32, _rknn_tensor_type_RKNN_TENSOR_INT16,
    _rknn_tensor_type_RKNN_TENSOR_INT32, _rknn_tensor_type_RKNN_TENSOR_INT64,
    _rknn_tensor_type_RKNN_TENSOR_INT8, _rknn_tensor_type_RKNN_TENSOR_TYPE_MAX,
    _rknn_tensor_type_RKNN_TENSOR_UINT16, _rknn_tensor_type_RKNN_TENSOR_UINT32,
    _rknn_tensor_type_RKNN_TENSOR_UINT8,
};

use crate::{queries::RknnQuery, QueryObject};

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

pub struct RknnInputTensorAttribute<const T: usize>(pub RknnTensorAttribute);

impl<const T: usize> QueryObject for RknnInputTensorAttribute<T> {
    type RknnPrimitiveType = _rknn_tensor_attr;

    fn primitive_init_value() -> Self::RknnPrimitiveType {
        let mut init = unsafe { std::mem::zeroed::<_rknn_tensor_attr>() };
        init.index = T as u32;
        init
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

pub struct RknnOutputTensorAttribute<const T: usize>(pub RknnTensorAttribute);

impl<const T: usize> QueryObject for RknnOutputTensorAttribute<T> {
    type RknnPrimitiveType = _rknn_tensor_attr;

    fn primitive_init_value() -> Self::RknnPrimitiveType {
        let mut init = unsafe { std::mem::zeroed::<_rknn_tensor_attr>() };
        init.index = T as u32;
        init
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

#[derive(Debug, PartialEq, PartialOrd)]
pub enum RknnTensorQuantFormat {
    None,
    DynamicFixedPoint(i8),
    AffineScale(i32, f32),
}

#[allow(non_upper_case_globals)]
impl RknnTensorQuantFormat {
    fn from_spec(
        quant_type: _rknn_tensor_qnt_type,
        fixed_point: i8,
        zero_point: i32,
        scale: f32,
    ) -> Result<Self> {
        match quant_type {
            _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_NONE => Ok(Self::None),
            _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_DFP => Ok(Self::DynamicFixedPoint(fixed_point)),
            _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC => {
                Ok(Self::AffineScale(zero_point, scale))
            }
            _ => bail!("Unrecognized quantization type."),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd)]
#[repr(u32)]
pub enum RknnTensorFormat {
    NCHW = _rknn_tensor_format_RKNN_TENSOR_NCHW,
    NHWC = _rknn_tensor_format_RKNN_TENSOR_NHWC,
    NC1HWC2 = _rknn_tensor_format_RKNN_TENSOR_NC1HWC2,
    UNDEFINED = _rknn_tensor_format_RKNN_TENSOR_UNDEFINED,
    MAX = _rknn_tensor_format_RKNN_TENSOR_FORMAT_MAX,
}

#[derive(Debug, PartialEq, Eq, PartialOrd)]
#[repr(u32)]
pub enum RknnTensorType {
    F32 = _rknn_tensor_type_RKNN_TENSOR_FLOAT32,
    F16 = _rknn_tensor_type_RKNN_TENSOR_FLOAT16,
    I8 = _rknn_tensor_type_RKNN_TENSOR_INT8,
    U8 = _rknn_tensor_type_RKNN_TENSOR_UINT8,
    I16 = _rknn_tensor_type_RKNN_TENSOR_INT16,
    U16 = _rknn_tensor_type_RKNN_TENSOR_UINT16,
    I32 = _rknn_tensor_type_RKNN_TENSOR_INT32,
    U32 = _rknn_tensor_type_RKNN_TENSOR_UINT32,
    I64 = _rknn_tensor_type_RKNN_TENSOR_INT64,
    BOOL = _rknn_tensor_type_RKNN_TENSOR_BOOL,
    MAX = _rknn_tensor_type_RKNN_TENSOR_TYPE_MAX,
}
