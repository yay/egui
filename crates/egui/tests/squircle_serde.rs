#![cfg(feature = "persistence")]

use egui::epaint::{Color32, CornerShape, RectShape};
use egui::{Pos2, Rect, vec2};

#[test]
fn legacy_rectangles_inherit_default_and_squircles_round_trip() {
    let rect = Rect::from_min_size(Pos2::ZERO, vec2(100.0, 80.0));
    let round = RectShape::filled(rect, 8, Color32::WHITE);
    let serialized = ron::to_string(&round).unwrap();
    // Model a persisted rectangle written before the corner-shape field existed.
    let legacy = serialized.replace(",corner_shape:None", "");
    assert_ne!(serialized, legacy);
    assert_eq!(ron::from_str::<RectShape>(&legacy).unwrap(), round);

    let squircle = round.with_corner_shape(CornerShape::Squircle);
    let serialized = ron::to_string(&squircle).unwrap();
    assert_eq!(ron::from_str::<RectShape>(&serialized).unwrap(), squircle);
}

#[test]
fn legacy_tessellation_options_default_to_round() {
    let options = egui::epaint::TessellationOptions::default();
    let serialized = ron::to_string(&options).unwrap();
    let legacy = serialized.replace(",corner_shape:Round", "");
    assert_ne!(serialized, legacy);
    assert_eq!(
        ron::from_str::<egui::epaint::TessellationOptions>(&legacy).unwrap(),
        options
    );
}
