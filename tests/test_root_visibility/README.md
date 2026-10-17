# Root visibility regression probe

This native probe covers the first-frame reveal guard proposed in
[egui #8459](https://github.com/emilk/egui/issues/8459#issuecomment-6154254324).
It starts real windows, verifies their visibility through winit, then closes itself.
It does not load application data or require interaction.

```sh
cargo run --locked --release -p test_root_visibility -- --renderer wgpu --initial hidden
cargo run --locked --release -p test_root_visibility -- --renderer glow --initial hidden
```

Repeat with `--initial default` and `--initial visible` to verify the ordinary first-frame
reveal remains intact. Add `--no-child` to the hidden case to exercise a tray-style root
with no visible windows. The default case creates a deferred child at a separate position
so that the controller does not occlude it during explicit visibility transitions.

The probe verifies that the root is hidden during app creation; an initially hidden root
stays hidden through its first frames; explicit show/hide/show/hide commands work; root
and deferred-child callbacks keep progressing while the root is hidden; and closing a
hidden root exits the native event loop. A 20-second watchdog fails stalled runs.

On 2026-10-17, all eight combinations passed on macOS: WGPU and Glow, each with hidden,
default, explicitly visible, and hidden-without-child startup. Restoring the unmodified
0.37.0 automatic reveal made the hidden-root assertion fail, confirming regression coverage.
Windows and Linux native runs remain pending. This is callback-level visibility sampling,
not an OS event trace that could detect every sub-frame show/hide transition.

The production change only suppresses the automatic reveal for `with_visible(false)`.
It keeps a native root window and rendering surface; it does not implement a windowless
root or change eframe's background repaint scheduling or restoration behavior.
