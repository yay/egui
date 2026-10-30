/// The convex curve used for the corners of a [`crate::RectShape`].
///
/// This selects the outline only. Corner radii, stroke placement, feathering, and texturing
/// retain their usual semantics. Circular corners remain the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum CornerShape {
    /// Circular arcs, using the existing rounded-rectangle tessellation path.
    #[default]
    Round,

    /// Convex superellipse arcs with exponent four: `x⁴ + y⁴ = r⁴`.
    ///
    /// Each arc occupies the same corner square as a circular arc of the given radius.
    /// A zero radius still produces a sharp corner. No concave or custom exponents are supported.
    /// As with round corners, tessellation may enlarge radii to fit the stroke. If the rectangle
    /// is too small for a nonintersecting inset, affected corners use circular arcs instead.
    Squircle,
}
