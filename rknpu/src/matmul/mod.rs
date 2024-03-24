use std::{
    ffi::c_void,
    ptr::{copy_nonoverlapping, NonNull},
};

use anyhow::{anyhow, Result};
use rknpu_sys::{
    _rknn_matmul_type, _rknn_matmul_type_RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
    _rknn_matmul_type_RKNN_INT4_MM_INT4_TO_INT16, _rknn_matmul_type_RKNN_INT8_MM_INT8_TO_INT32,
    rknn_create_mem, rknn_matmul_create, rknn_matmul_ctx, rknn_matmul_info, rknn_matmul_io_attr,
    rknn_matmul_run, rknn_matmul_set_io_mem, rknn_tensor_mem,
};

use crate::error::check_result;

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum RknnMatmulType {
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = _rknn_matmul_type_RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
    RKNN_INT8_MM_INT8_TO_INT32 = _rknn_matmul_type_RKNN_INT8_MM_INT8_TO_INT32,
    RKNN_INT4_MM_INT4_TO_INT16 = _rknn_matmul_type_RKNN_INT4_MM_INT4_TO_INT16,
}

#[derive(Debug, Clone)]
pub struct RknnMatmulInfo {
    m: usize,
    k: usize,
    n: usize,
    mm_type: RknnMatmulType,
    b_native_layout: bool,
    ac_native_layout: bool,
}

impl RknnMatmulInfo {
    pub fn new(m: usize, k: usize, n: usize, mm_type: RknnMatmulType) -> Self {
        Self {
            m,
            k,
            n,
            mm_type,
            b_native_layout: true,
            ac_native_layout: true,
        }
    }
}

impl From<RknnMatmulInfo> for rknn_matmul_info {
    fn from(v: RknnMatmulInfo) -> Self {
        Self {
            M: v.m as i32,
            K: v.k as i32,
            N: v.n as i32,
            type_: v.mm_type as _rknn_matmul_type,
            B_layout: v.b_native_layout as i32,
            AC_layout: v.ac_native_layout as i32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RknnMatmul {
    ctx_ptr: rknn_matmul_ctx,
    infos: RknnMatmulInfo,
    io_attr: rknn_matmul_io_attr,
    a_buffer: NonNull<rknn_tensor_mem>,
    b_buffer: NonNull<rknn_tensor_mem>,
    c_buffer: NonNull<rknn_tensor_mem>,
}

impl RknnMatmul {
    pub fn new(infos: RknnMatmulInfo) -> Result<Self> {
        let mut ctx_ptr = unsafe { std::mem::zeroed::<rknn_matmul_ctx>() };
        let mut io_attr = unsafe { std::mem::zeroed::<rknn_matmul_io_attr>() };
        let mut rknn_input_infos: rknn_matmul_info = infos.clone().into();
        let ret = unsafe { rknn_matmul_create(&mut ctx_ptr, &mut rknn_input_infos, &mut io_attr) };
        check_result(ret)?;

        let (a_buffer, b_buffer, c_buffer) = unsafe {
            (
                NonNull::new(rknn_create_mem(ctx_ptr, io_attr.A.size))
                    .ok_or(anyhow!("Could not create memory buffer for matrix A"))?,
                NonNull::new(rknn_create_mem(ctx_ptr, io_attr.B.size))
                    .ok_or(anyhow!("Could not create memory buffer for matrix B"))?,
                NonNull::new(rknn_create_mem(ctx_ptr, io_attr.C.size))
                    .ok_or(anyhow!("Could not create memory buffer for matrix C"))?,
            )
        };
        Ok(Self {
            ctx_ptr,
            infos,
            io_attr,
            a_buffer,
            b_buffer,
            c_buffer,
        })
    }

    pub fn run(&mut self, a: &[u8], b: &[u8], c: &[u8]) -> Result<()> {
        unsafe {
            copy_nonoverlapping(
                a.as_ptr() as *mut c_void,
                (*self.a_buffer.as_ptr()).virt_addr,
                (*self.a_buffer.as_ptr()).size as usize,
            );
            copy_nonoverlapping(
                b.as_ptr() as *mut c_void,
                (*self.b_buffer.as_ptr()).virt_addr,
                (*self.b_buffer.as_ptr()).size as usize,
            );
            check_result(rknn_matmul_set_io_mem(
                self.ctx_ptr,
                self.a_buffer.as_mut(),
                &mut self.io_attr.A,
            ))?;
            check_result(rknn_matmul_set_io_mem(
                self.ctx_ptr,
                self.b_buffer.as_mut(),
                &mut self.io_attr.B,
            ))?;
            check_result(rknn_matmul_set_io_mem(
                self.ctx_ptr,
                self.c_buffer.as_mut(),
                &mut self.io_attr.C,
            ))?;
        };

        let ret = unsafe { rknn_matmul_run(self.ctx_ptr) };
        check_result(ret)?;

        unsafe {
            copy_nonoverlapping(
                (*self.b_buffer.as_ptr()).virt_addr,
                c.as_ptr() as *mut c_void,
                (*self.b_buffer.as_ptr()).size as usize,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::ops::{AddAssign, Mul};

    use half::f16;

    fn matmul<T: Mul<T, Output = T> + AddAssign<T> + Default + Copy + Clone>(
        a: &[T],
        b: &[T],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<T> {
        let mut c = vec![T::default(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = T::default();
                for k_idx in 0..k {
                    acc += a[i * k + k_idx] * b[i * n + j]
                }
                c[i * n + j] = acc;
            }
        }

        c
    }

    #[test]
    fn test_matmu_ref() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let res = matmul(a.as_slice(), b.as_slice(), 2, 2, 2);
        dbg!(res);
    }

    #[test]
    fn test_simple_matmul() {
        let a = (0..512 * 512)
            .map(|x| f16::from_f32(x as f32))
            .collect::<Vec<f16>>();
        let b = (0..512 * 512)
            .map(|x| f16::from_f32(x as f32))
            .collect::<Vec<f16>>();
    }
}
