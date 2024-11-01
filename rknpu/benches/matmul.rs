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
    b_quant_type: QuantType,
    ac_layout: bool,
    ac_quant_type: QuantType,
    group_size: Option<usize>,
) {
    let mut group = c.benchmark_group(format!("{}x{}x{}", m, k, n));
    group.throughput(Throughput::Elements((m * k * n) as _));

    let infos = RknnMatmulInfo::new(
        m,
        k,
        n,
        t.clone(),
        b_layout,
        b_quant_type,
        ac_layout,
        ac_quant_type,
        0,
        group_size,
    );
    let mut rknn_matmul = RknnMatmul::new(infos).unwrap();

    // Initialize matrices A and B based on the RknnMatmulType
    match t {
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
            rknn_matmul.set_inputs(raw_a, raw_b).unwrap();
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
            rknn_matmul.set_inputs(raw_a, raw_b).unwrap();
        }
        RknnMatmulType::RKNN_INT4_MM_INT4_TO_INT16 => {
            // Adjust buffer sizes for int4
            let a = vec![0_u8; (m * k) / 2]; // Each u8 stores two int4 values
            let b = vec![0_u8; (k * n) / 2]; // Each u8 stores two int4 values
            let raw_a = unsafe { std::slice::from_raw_parts(a.as_ptr(), a.len()) };
            let raw_b = unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) };
            rknn_matmul.set_inputs(raw_a, raw_b).unwrap();
        }
    };

    group.bench_function(label, |b| {
        b.iter(|| {
            rknn_matmul.exec().unwrap();
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
            QuantType::Layer,
            true,
            QuantType::Layer,
            None,
        ),
        (
            "int8",
            RknnMatmulType::RKNN_INT8_MM_INT8_TO_INT32,
            true,
            QuantType::Layer,
            true,
            QuantType::Layer,
            None,
        ),
        (
            "int4",
            RknnMatmulType::RKNN_INT4_MM_INT4_TO_INT16,
            true,
            QuantType::Layer,
            true,
            QuantType::Layer,
            None,
        ),
    ];

    for (label, matmul_type, b_layout, b_quant_type, ac_layout, ac_quant_type, group_size) in
        matmul_variants.into_iter()
    {
        matmul_benchmark(
            c,
            m,
            k,
            n,
            matmul_type,
            label,
            b_layout,
            b_quant_type,
            ac_layout,
            ac_quant_type,
            group_size,
        );
    }
}

fn matmul(c: &mut Criterion) {
    let m_values = [4, 8, 64, 128, 256];
    let k_values = [256, 512, 1024, 2048, 4096];
    let n_values = [256, 512, 1024, 2048, 4096];

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
