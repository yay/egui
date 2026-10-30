use core::{hint::black_box, time::Duration};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use epaint::{
    Color32, CornerShape, Mesh, Pos2, Rect, RectShape, Stroke, StrokeKind, TessellationOptions,
    Tessellator, TextureId, pos2, vec2,
};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Measure steady-state rectangle tessellation with reusable mesh and scratch buffers.
/// All cases use identical geometry, colors, scale, and fill/stroke/texture settings.
fn rectangles(c: &mut Criterion) {
    let mut group = c.benchmark_group("rectangles");
    group.throughput(Throughput::Elements(1));
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);
    let fill = Color32::from_rgba_premultiplied(64, 64, 64, 128);
    let rect = Rect::from_min_size(Pos2::ZERO, vec2(100.0, 100.0));
    let fill_only = RectShape::filled(rect, 8, fill);
    let filled_stroke = RectShape::new(rect, 8, fill, Stroke::new(4.0, fill), StrokeKind::Inside);
    let textured = fill_only.clone().with_texture(
        TextureId::User(1),
        Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
    );
    for (name, shape) in [
        ("fill", fill_only),
        ("filled_stroke", filled_stroke),
        ("textured", textured),
    ] {
        for (curve, shape, corner_shape) in [
            ("round", shape.clone(), CornerShape::Round),
            ("global_squircle", shape.clone(), CornerShape::Squircle),
            (
                "squircle",
                shape.with_corner_shape(CornerShape::Squircle),
                CornerShape::Round,
            ),
        ] {
            let options = TessellationOptions {
                round_rects_to_pixels: false,
                corner_shape,
                ..Default::default()
            };
            let mut tessellator = Tessellator::new(1.0, options, [1024, 1024], vec![]);
            let mut mesh = Mesh::default();
            tessellator.tessellate_rect(&shape, &mut mesh);
            group.bench_function(format!("{curve}/{name}"), |b| {
                b.iter(|| {
                    mesh.clear();
                    tessellator.tessellate_rect(black_box(&shape), &mut mesh);
                    black_box(&mesh);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, rectangles);
criterion_main!(benches);
