use criterion::*;
use half::f16;
use rknpu::matmul::*;

fn matmul_benchmark(
    c: &mut Criterion,
    m: usize,
    k: usize,
    n: usize,
    t: RknnMatmulType,
    label: &str,
    b_layout: bool,
    ac_layout: bool,
) {
    let mut group = c.benchmark_group(format!("{}x{}x{} {}", m, k, n, label));
    group.throughput(Throughput::Elements((m * k * n) as _));

    let infos = RknnMatmulInfo::new(m, k, n, t.clone(), b_layout, ac_layout);
    let mut rknn_matmul = RknnMatmul::new(infos).unwrap();

    // Initialize matrices A and B based on the RknnMatmulType
    let (raw_a, raw_b, mut raw_c): (&[u8], &[u8], Vec<u8>) = match t {
        RknnMatmulType::RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 => {
            let a = vec![f16::from_f32(0.0); m * k];
            let b = vec![f16::from_f32(0.0); k * n];
            let raw_a = unsafe {
                std::slice::from_raw_parts(
                    a.as_ptr() as *const u8,
                    a.len() * std::mem::size_of::<f16>(),
                )
            };
            let raw_b = unsafe {
                std::slice::from_raw_parts(
                    b.as_ptr() as *const u8,
                    b.len() * std::mem::size_of::<f16>(),
                )
            };
            let raw_c = vec![0; m * n * std::mem::size_of::<f32>()];
            (raw_a, raw_b, raw_c)
        }
        RknnMatmulType::RKNN_INT8_MM_INT8_TO_INT32 => {
            let a = vec![0_i8; m * k];
            let b = vec![0_i8; k * n];
            let raw_a = unsafe {
                std::slice::from_raw_parts(
                    a.as_ptr() as *const u8,
                    a.len() * std::mem::size_of::<i8>(),
                )
            };
            let raw_b = unsafe {
                std::slice::from_raw_parts(
                    b.as_ptr() as *const u8,
                    b.len() * std::mem::size_of::<i8>(),
                )
            };
            let raw_c = vec![0; m * n * std::mem::size_of::<i32>()];
            (raw_a, raw_b, raw_c)
        }
        RknnMatmulType::RKNN_INT4_MM_INT4_TO_INT16 => {
            // Adjust buffer sizes for int4
            let a = vec![0_u8; (m * k) / 2]; // Each u8 stores two int4 values
            let b = vec![0_u8; (k * n) / 2]; // Each u8 stores two int4 values
            let raw_a = unsafe { std::slice::from_raw_parts(a.as_ptr(), a.len()) };
            let raw_b = unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) };
            let raw_c = vec![0; m * n * std::mem::size_of::<i16>()];
            (raw_a, raw_b, raw_c)
        }
    };

    group.bench_function(label, |b| {
        b.iter(|| {
            rknn_matmul.run(raw_a, raw_b, raw_c.as_mut_slice()).unwrap();
        })
    });

    group.finish();
}

fn bench_method(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let matmul_variants = [
        (
            "float16",
            RknnMatmulType::RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            true,
            true,
        ),
        (
            "int8",
            RknnMatmulType::RKNN_INT8_MM_INT8_TO_INT32,
            true,
            true,
        ),
        (
            "int4",
            RknnMatmulType::RKNN_INT4_MM_INT4_TO_INT16,
            true,
            true,
        ),
    ];

    for (label, matmul_type, b_layout, ac_layout) in matmul_variants.into_iter() {
        matmul_benchmark(c, m, k, n, matmul_type, label, b_layout, ac_layout);
    }
}

fn matmul(c: &mut Criterion) {
    let m_values = [128, 256, 512];
    let k_values = [256, 512, 1024, 2048];
    let n_values = [256, 512, 1024, 2048];

    for &m in &m_values {
        for &k in &k_values {
            for &n in &n_values {
                bench_method(c, m, k, n);
            }
        }
    }
}
criterion_group!(benches, matmul);
criterion_main!(benches);
