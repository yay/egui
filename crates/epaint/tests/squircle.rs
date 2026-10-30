use epaint::{
    Color32, CornerRadius, CornerRadiusF32, CornerShape, Mesh, Pos2, Rect, RectShape, RoundedRect,
    Shape, Stroke, StrokeKind, TessellationOptions, Tessellator, TextureId, Vec2, pos2,
    tessellator::path, vec2,
};

/// Tessellate a rectangle with explicit antialiasing and scale, without pixel snapping.
fn mesh_for(shape: &RectShape, feathering: bool, pixels_per_point: f32) -> Mesh {
    let options = TessellationOptions {
        feathering,
        round_rects_to_pixels: false,
        ..Default::default()
    };
    let mut tessellator = Tessellator::new(pixels_per_point, options, [1024, 1024], vec![]);
    let mut mesh = Mesh::default();
    tessellator.tessellate_rect(shape, &mut mesh);
    mesh
}

/// Signed two-dimensional cross product, positive for clockwise turns in screen coordinates.
fn cross(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

/// Sample untextured triangle alpha using the renderer's source-over blending order.
/// Callers avoid triangle edges; degenerate triangles have no coverage.
fn sample_alpha(mesh: &Mesh, point: Pos2) -> f32 {
    let mut alpha = 0.0;
    for indices in mesh.indices.chunks_exact(3) {
        let [a, b, c] =
            [indices[0], indices[1], indices[2]].map(|index| mesh.vertices[index as usize]);
        let area = cross(b.pos - a.pos, c.pos - a.pos);
        if area.abs() < 1e-8 {
            continue;
        }
        let u = cross(b.pos - point, c.pos - point) / area;
        let v = cross(c.pos - point, a.pos - point) / area;
        let w = 1.0 - u - v;
        if u > 0.0 && v > 0.0 && w > 0.0 {
            let source = (u * f32::from(a.color.a())
                + v * f32::from(b.color.a())
                + w * f32::from(c.color.a()))
                / 255.0;
            alpha = source + alpha * (1.0 - source);
        }
    }
    alpha
}

#[test]
fn compact_layout_and_default_are_preserved() {
    assert_eq!(size_of::<CornerRadius>(), 4);
    assert_eq!(size_of::<CornerRadiusF32>(), 16);
    // The opt-in discriminator fits in existing padding on 64-bit targets.
    if cfg!(target_pointer_width = "64") {
        assert_eq!(size_of::<RectShape>(), 56);
        assert_eq!(size_of::<Shape>(), 64);
    }
    let shape = RectShape::filled(
        Rect::from_min_size(Pos2::ZERO, vec2(100.0, 80.0)),
        12,
        Color32::WHITE,
    );
    assert_eq!(shape.corner_shape, None);
    assert_eq!(
        mesh_for(&shape, true, 2.0),
        mesh_for(&shape.with_corner_shape(CornerShape::Round), true, 2.0)
    );
}

#[test]
fn rectangles_inherit_the_global_default_but_explicit_overrides_win() {
    let shape = RectShape::filled(
        Rect::from_min_size(Pos2::ZERO, vec2(100.0, 80.0)),
        12,
        Color32::WHITE,
    );
    let render = |shape: &RectShape, corner_shape| {
        let options = TessellationOptions {
            corner_shape,
            ..Default::default()
        };
        let mut tessellator = Tessellator::new(1.0, options, [1024, 1024], vec![]);
        let mut mesh = Mesh::default();
        tessellator.tessellate_rect(shape, &mut mesh);
        mesh
    };
    let round = render(&shape, CornerShape::Round);
    let squircle = render(&shape, CornerShape::Squircle);
    assert_ne!(round, squircle);
    assert_eq!(
        render(
            &shape.clone().with_corner_shape(CornerShape::Round),
            CornerShape::Squircle
        ),
        round,
    );
    assert_eq!(
        render(
            &shape.clone().with_corner_shape(CornerShape::Squircle),
            CornerShape::Round
        ),
        squircle,
    );

    let sharp = RectShape::filled(shape.rect, 0, Color32::WHITE);
    assert_eq!(
        render(&sharp, CornerShape::Round),
        render(&sharp, CornerShape::Squircle)
    );
}

#[test]
fn global_corner_shape_does_not_change_circles() {
    let render = |corner_shape| {
        let options = TessellationOptions {
            corner_shape,
            ..Default::default()
        };
        let mut tessellator = Tessellator::new(1.0, options, [1024, 1024], vec![]);
        let mut mesh = Mesh::default();
        tessellator.tessellate_shape(
            Shape::circle_filled(pos2(20.0, 20.0), 10.0, Color32::WHITE),
            &mut mesh,
        );
        mesh
    };
    assert_eq!(render(CornerShape::Round), render(CornerShape::Squircle));
}

#[test]
fn squircle_vertices_follow_exponent_four() {
    let rect = Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.0));
    let mut points = Vec::new();
    let radius = 20.0;
    path::squircle_rectangle(&mut points, RoundedRect::new(rect, radius));
    for p in &points {
        let center = pos2(
            if p.x < 50.0 { 20.0 } else { 80.0 },
            if p.y < 50.0 { 20.0 } else { 80.0 },
        );
        let offset = (*p - center) / radius;
        let equation = offset.x.powi(4) + offset.y.powi(4);
        assert!((equation - 1.0).abs() < 1e-5, "{p:?}: {equation}");
    }
    let diagonal = 80.0 + radius * 0.5_f32.powf(0.25);
    assert!(
        points
            .iter()
            .any(|p| p.distance(pos2(diagonal, diagonal)) < 1e-4)
    );
    for endpoint in [pos2(100.0, 80.0), pos2(80.0, 100.0), pos2(0.0, 20.0)] {
        assert!(points.contains(&endpoint));
    }
}

#[test]
fn squircle_outlines_are_convex_and_have_no_duplicate_neighbors() {
    let mut points = Vec::new();
    for size in [vec2(100.0, 100.0), vec2(100.0, 7.0), vec2(0.5, 40.0)] {
        let rect = Rect::from_min_size(pos2(0.25, 0.75), size);
        for radius in [0.0, 0.1, 2.0, 5.0, 8.0, 18.0, 50.0, 255.0] {
            for cr in [
                CornerRadiusF32::same(radius),
                CornerRadiusF32 {
                    nw: radius,
                    ne: 0.0,
                    sw: radius * 0.5,
                    se: radius * 0.75,
                },
            ] {
                path::squircle_rectangle(&mut points, RoundedRect::new(rect, cr));
                assert!(points.len() >= 4);
                for i in 0..points.len() {
                    let a = points[i];
                    let b = points[(i + 1) % points.len()];
                    let c = points[(i + 2) % points.len()];
                    assert_ne!(a, b, "size={size:?}, radius={radius}");
                    assert!(rect.contains(a), "{a:?} outside {rect:?}");
                    assert!(cross(b - a, c - b) >= -1e-5, "nonconvex: {a:?} {b:?} {c:?}");
                }
            }
        }
    }
}

#[test]
fn zero_radii_keep_the_exact_circular_mesh() {
    let shape = RectShape::new(
        Rect::from_min_size(Pos2::ZERO, vec2(100.0, 80.0)),
        0,
        Color32::WHITE,
        Stroke::new(2.0, Color32::BLACK),
        StrokeKind::Inside,
    );
    for feathering in [false, true] {
        assert_eq!(
            mesh_for(&shape, feathering, 1.0),
            mesh_for(
                &shape.clone().with_corner_shape(CornerShape::Squircle),
                feathering,
                1.0
            )
        );
    }
}

#[test]
fn translucent_fill_and_stroke_do_not_overlap() {
    let color = Color32::from_rgba_premultiplied(64, 64, 64, 128);
    for (corner_shape, radius, width) in [
        (CornerShape::Round, 8, 4.0_f32),
        (CornerShape::Round, 16, 4.0),
        (CornerShape::Squircle, 1, 4.0),
        (CornerShape::Squircle, 8, 4.0),
        (CornerShape::Squircle, 16, 4.0),
        (CornerShape::Squircle, 8, 20.0),
        (CornerShape::Squircle, 255, 20.0),
    ] {
        for kind in [StrokeKind::Inside, StrokeKind::Middle, StrokeKind::Outside] {
            for feathering in [false, true] {
                let shape = RectShape::new(
                    Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.0)),
                    radius,
                    color,
                    Stroke::new(width, color),
                    kind,
                )
                .with_corner_shape(corner_shape);
                let mesh = mesh_for(&shape, feathering, 2.0);
                // Includes the border, its transition to the fill, and the curved corners.
                for y in 0..100 {
                    for x in 0..100 {
                        let alpha = sample_alpha(&mesh, pos2(x as f32 + 0.31, y as f32 + 0.67));
                        assert!(
                            alpha <= 128.0 / 255.0 + 1e-5,
                            concat!("overlap: {:?}, {:?}, aa={}, ", "r={}, w={}, ({}, {})"),
                            corner_shape,
                            kind,
                            feathering,
                            radius,
                            width,
                            x,
                            y
                        );
                    }
                }
                let border_x = match kind {
                    StrokeKind::Inside => 0.325 * width,
                    StrokeKind::Middle => -0.175 * width,
                    StrokeKind::Outside => -0.675 * width,
                };
                for point in [pos2(border_x, 40.7), pos2(50.3, 50.7)] {
                    assert!((sample_alpha(&mesh, point) - 128.0 / 255.0).abs() < 1e-5);
                }
            }
        }
    }
}

#[test]
fn meshes_remain_finite_with_clamping_rotation_blur_and_textures() {
    for radius in [0, 1, 8, 50, 255] {
        for width in [0.0_f32, 0.25, 1.0, 4.0, 30.0] {
            for kind in [StrokeKind::Inside, StrokeKind::Middle, StrokeKind::Outside] {
                for angle in [0.0, 0.7] {
                    for blur in [0.0, 8.0] {
                        for feathering in [false, true] {
                            for pixels_per_point in [1.0, 1.5, 2.0] {
                                let shape = RectShape::new(
                                    Rect::from_min_size(pos2(0.3, 0.7), vec2(60.0, 45.0)),
                                    CornerRadius {
                                        nw: radius,
                                        ne: 0,
                                        sw: radius / 2,
                                        se: radius,
                                    },
                                    Color32::WHITE,
                                    Stroke::new(width, Color32::BLACK),
                                    kind,
                                )
                                .with_corner_shape(CornerShape::Squircle)
                                .with_angle(angle)
                                .with_blur_width(blur);
                                // Solid strokes use the font atlas; test texture fills separately.
                                let textured = RectShape {
                                    stroke: Stroke::NONE,
                                    ..shape.clone()
                                };
                                for shape in [
                                    shape,
                                    textured.with_texture(
                                        TextureId::User(7),
                                        Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
                                    ),
                                ] {
                                    let mesh = mesh_for(&shape, feathering, pixels_per_point);
                                    assert!(mesh.is_valid());
                                    assert!(!mesh.vertices.is_empty());
                                    assert!(mesh.vertices.iter().all(|v| v.pos.x.is_finite()
                                        && v.pos.y.is_finite()
                                        && v.uv.x.is_finite()
                                        && v.uv.y.is_finite()));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn textured_squircle_keeps_uv_mapping_and_texture() {
    let rect = Rect::from_min_size(pos2(10.0, 20.0), vec2(100.0, 80.0));
    let shape = RectShape::filled(rect, 20, Color32::WHITE)
        .with_corner_shape(CornerShape::Squircle)
        .with_texture(
            TextureId::User(7),
            Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
        );
    let mesh = mesh_for(&shape, false, 1.0);
    assert_eq!(mesh.texture_id, TextureId::User(7));
    for vertex in mesh.vertices {
        let expected = (vertex.pos - rect.min) / rect.size();
        assert!((vertex.uv.to_vec2() - expected).length() < 1e-6);
    }
}
