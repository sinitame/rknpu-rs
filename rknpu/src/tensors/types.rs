use anyhow::{bail, Result};
use rknpu_sys::{
    _rknn_tensor_format_RKNN_TENSOR_FORMAT_MAX, _rknn_tensor_format_RKNN_TENSOR_NC1HWC2,
    _rknn_tensor_format_RKNN_TENSOR_NCHW, _rknn_tensor_format_RKNN_TENSOR_NHWC,
    _rknn_tensor_format_RKNN_TENSOR_UNDEFINED, _rknn_tensor_qnt_type,
    _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC,
    _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_DFP, _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_NONE,
    _rknn_tensor_type_RKNN_TENSOR_BOOL, _rknn_tensor_type_RKNN_TENSOR_FLOAT16,
    _rknn_tensor_type_RKNN_TENSOR_FLOAT32, _rknn_tensor_type_RKNN_TENSOR_INT16,
    _rknn_tensor_type_RKNN_TENSOR_INT32, _rknn_tensor_type_RKNN_TENSOR_INT64,
    _rknn_tensor_type_RKNN_TENSOR_INT8, _rknn_tensor_type_RKNN_TENSOR_TYPE_MAX,
    _rknn_tensor_type_RKNN_TENSOR_UINT16, _rknn_tensor_type_RKNN_TENSOR_UINT32,
    _rknn_tensor_type_RKNN_TENSOR_UINT8,
};

#[derive(Debug, PartialEq, PartialOrd)]
pub enum RknnTensorQuantFormat {
    None,
    DynamicFixedPoint(i8),
    AffineScale(i32, f32),
}

impl RknnTensorQuantFormat {
    pub fn zp_scale(&self) -> Option<(i32, f32)> {
        match self {
            Self::None => None,
            Self::AffineScale(zp, scale) => Some((*zp, *scale)),
            Self::DynamicFixedPoint(_) => None,
        }
    }
}

#[allow(non_upper_case_globals)]
impl RknnTensorQuantFormat {
    pub fn from_spec(
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
