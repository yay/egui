use egui::{Color32, CornerShape, Frame, Pos2, Rect, Shape, vec2};

/// Return clipped meshes so assertions compare actual geometry and color/texture data.
/// These tests only paint built-in shapes, so a callback primitive is an error.
fn tessellate(
    ctx: &egui::Context,
    shapes: Vec<egui::epaint::ClippedShape>,
    pixels_per_point: f32,
) -> Vec<(Rect, egui::Mesh)> {
    ctx.tessellate(shapes, pixels_per_point)
        .into_iter()
        .map(|primitive| {
            let egui::epaint::Primitive::Mesh(mesh) = primitive.primitive else {
                panic!("Unexpected paint callback");
            };
            (primitive.clip_rect, mesh)
        })
        .collect()
}

#[test]
fn context_default_reaches_standard_widget_and_frame_backgrounds() {
    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
        Frame::group(ui.style()).show(ui, |ui| {
            let _ = ui.button("Inherited corners");
        });
    });
    output.textures_delta.clear();
    let mut rectangles: Vec<_> = output.shapes.into_iter().filter(|shape| {
        matches!(&shape.shape, Shape::Rect(rect) if rect.corner_radius != egui::CornerRadius::ZERO)
    }).collect();
    assert!(
        rectangles.len() >= 2,
        "Expected the group frame and button background"
    );
    assert!(
        rectangles.iter().all(|shape| {
            matches!(&shape.shape, Shape::Rect(rect) if rect.corner_shape.is_none())
        })
    );

    let round = tessellate(&ctx, rectangles.clone(), output.pixels_per_point);
    ctx.tessellation_options_mut(|options| options.corner_shape = CornerShape::Squircle);
    let squircle = tessellate(&ctx, rectangles.clone(), output.pixels_per_point);
    assert_ne!(round, squircle);

    for shape in &mut rectangles {
        if let Shape::Rect(rect) = &mut shape.shape {
            rect.corner_shape = Some(CornerShape::Round);
        }
    }
    assert_eq!(tessellate(&ctx, rectangles, output.pixels_per_point), round);
}

#[test]
fn window_frame_and_shadow_inherit_the_context_default() {
    let ctx = egui::Context::default();
    ctx.run_ui(egui::RawInput::default(), |_| {})
        .textures_delta
        .clear();
    let frame = Frame::window(&ctx.style_of(egui::Theme::Dark)).fill(Color32::GRAY);
    let shape = frame.paint(Rect::from_min_size(Pos2::ZERO, vec2(100.0, 80.0)));
    let Shape::Vec(parts) = &shape else {
        panic!("Expected a window frame and shadow");
    };
    assert_eq!(parts.len(), 2);
    assert!(
        parts
            .iter()
            .all(|shape| { matches!(shape, Shape::Rect(rect) if rect.corner_shape.is_none()) })
    );
    let shapes = vec![egui::epaint::ClippedShape::new(Rect::EVERYTHING, shape)];
    let round = tessellate(&ctx, shapes.clone(), 1.0);
    ctx.tessellation_options_mut(|options| options.corner_shape = CornerShape::Squircle);
    assert_ne!(round, tessellate(&ctx, shapes, 1.0));
}
