use anyhow::Result;
use rknpu_sys::{
    _rknn_query_cmd_RKNN_QUERY_INPUT_ATTR, _rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
    _rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR, _rknn_query_cmd_RKNN_QUERY_PERF_DETAIL,
    _rknn_query_cmd_RKNN_QUERY_PERF_RUN, _rknn_query_cmd_RKNN_QUERY_SDK_VERSION,
};

pub trait QueryObject {
    type RknnPrimitiveType;
    fn primitive_init_value() -> Self::RknnPrimitiveType;
    fn query_flag() -> RknnQuery;
    fn from_primitive_type(value: Self::RknnPrimitiveType) -> Result<Self>
    where
        Self: Sized;
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum RknnQuery {
    // Query the number of input & output tensor.
    RKNN_QUERY_IN_OUT_NUM = _rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
    // Query the attribute of input tensor.
    RKNN_QUERY_INPUT_ATTR = _rknn_query_cmd_RKNN_QUERY_INPUT_ATTR,
    // Query the attribute of output tensor.
    RKNN_QUERY_OUTPUT_ATTR = _rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR,
    // Query the detail performance, need set RKNN_FLAG_COLLECT_PERF_MASK
    // when call rknn_init, this query needs to be valid after rknn_outputs_get.
    RKNN_QUERY_PERF_DETAIL = _rknn_query_cmd_RKNN_QUERY_PERF_DETAIL,
    // Query the time of run, this query needs to be valid after rknn_outputs_get.
    RKNN_QUERY_PERF_RUN = _rknn_query_cmd_RKNN_QUERY_PERF_RUN,
    // Query the sdk & driver version
    RKNN_QUERY_SDK_VERSION = _rknn_query_cmd_RKNN_QUERY_SDK_VERSION,
    // ...
}
