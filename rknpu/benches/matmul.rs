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
    let mut group = c.benchmark_group(format!("{}x{}x{}", m, k, n));
    group.throughput(Throughput::Elements((m * k * n) as _));

    let infos = RknnMatmulInfo::new(m, k, n, t, b_layout, ac_layout);
    let mut rknn_matmul = RknnMatmul::new(infos).unwrap();

    // Initialize matrices A and B
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
    let mut raw_c: Vec<u8> = vec![0; m * n * std::mem::size_of::<f32>()]; // Adjusted size of raw_c to match m and n dimensions

    group.bench_function(label, |b| {
        b.iter(|| {
            rknn_matmul.run(raw_a, raw_b, raw_c.as_mut_slice()).unwrap();
        })
    });

    group.finish();
}

fn bench_method(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let matmul_variants = [
        //("rknn_matmul", RknnMatmulType::RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, false, false),
        (
            "rknn_matmul_opt",
            RknnMatmulType::RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            true,
            true,
        ),
    ];

    for (label, matmul_type, b_layout, ac_layout) in matmul_variants.into_iter() {
        matmul_benchmark(c, m, k, n, matmul_type, label, b_layout, ac_layout);
    }
}

fn matmul(c: &mut Criterion) {
    let sizes = [
        //(256, 64, 256),
        //(256, 128, 256),
        //(256, 256, 256),

        //(64, 256, 256),
        //(128, 256, 256),
        //(256, 256, 256),
        (256, 256, 64),
        (256, 256, 128),
        (256, 256, 256),
        (512, 512, 64),
        (512, 512, 128),
        (512, 512, 256),
        (1024, 1024, 64),
        (1024, 1024, 128),
        (1024, 1024, 256),
    ];

    for &(m, k, n) in &sizes {
        bench_method(c, m, k, n);
    }
}

criterion_group!(benches, matmul);
criterion_main!(benches);
