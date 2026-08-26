//! Native regression check; run from the egui fork with the sibling Steady winit fork:
//! ```sh
//! cargo run -p eframe --release --example windowless_lifecycle --features glow \
//!   --config 'patch.crates-io.winit.path="../winit"' -- glow
//! ```
//! Replace the final `glow` with `wgpu` to check the other renderer.
//! Each run opens small windows, checks repaint/lifecycle behavior, then closes them.

#[cfg(any(target_os = "macos", target_os = "windows"))]
mod native {
    #![expect(
        clippy::print_stdout,
        reason = "standalone regression check reports results to stdout"
    )]

    use eframe::egui;
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst},
        },
        thread,
        time::Duration,
    };

    #[derive(Clone, Copy, Debug)]
    enum Mode {
        Immediate,
        NestedImmediate,
        Deferred,
        Windowed,
    }

    /// Coordinates the UI and its worker. Counts are UI callbacks, not physical presentations.
    #[derive(Default)]
    struct State {
        show: AtomicBool,
        callbacks: AtomicUsize,
        cancel_next_close: AtomicBool,
        cancelled_closes: AtomicUsize,
        focused: AtomicBool,
    }

    struct Probe {
        mode: Mode,
        state: Arc<State>,
    }

    fn child_id() -> egui::ViewportId {
        egui::ViewportId::from_hash_of("windowless lifecycle leaf")
    }

    fn builder() -> egui::ViewportBuilder {
        egui::ViewportBuilder::default()
            .with_title("eframe windowless lifecycle check")
            .with_inner_size([240.0, 80.0])
            .with_active(false)
    }

    impl eframe::App for Probe {
        fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
            let ctx = ui.ctx();
            if ctx.input(|i| i.viewport().close_requested())
                && self.state.cancel_next_close.swap(false, SeqCst)
            {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.state.cancelled_closes.fetch_add(1, SeqCst);
            }
            if matches!(self.mode, Mode::Windowed) {
                let window = frame
                    .winit_window()
                    .expect("windowed root has a native window");
                self.state.focused.fetch_or(window.has_focus(), SeqCst);
                self.state.callbacks.fetch_add(1, SeqCst);
                ui.label("Checking windowed root…");
                return;
            }
            assert!(
                frame.winit_window().is_none(),
                "logical root must not expose a native window"
            );
            if !self.state.show.load(SeqCst) {
                return;
            }

            let state = Arc::clone(&self.state);
            let callback = move |ui: &mut egui::Ui, _: egui::ViewportClass| {
                state.callbacks.fetch_add(1, SeqCst);
                ui.label("Checking native viewport lifecycle…");
            };
            match self.mode {
                Mode::Immediate => ctx.show_viewport_immediate(child_id(), builder(), callback),
                Mode::NestedImmediate => ctx.show_viewport_immediate(
                    egui::ViewportId::from_hash_of("windowless lifecycle parent"),
                    builder(),
                    |ui, _| {
                        ui.ctx()
                            .show_viewport_immediate(child_id(), builder(), &callback);
                    },
                ),
                Mode::Deferred => ctx.show_viewport_deferred(child_id(), builder(), callback),
                Mode::Windowed => unreachable!(),
            }
        }

        fn persist_egui_memory(&self) -> bool {
            false
        }
    }

    /// Drives checks off the event thread, collecting failures while always requesting shutdown.
    /// Delays allow native creation and its settling passes to finish before sampling callbacks.
    fn exercise(ctx: &egui::Context, state: &State, mode: Mode) -> Vec<String> {
        let mut failures = Vec::new();
        let mut check = |passed: bool, message: &str| {
            if !passed {
                failures.push(message.to_owned());
            }
        };
        let settle = || thread::sleep(Duration::from_millis(500));

        if matches!(mode, Mode::Windowed) {
            settle();
            ctx.request_repaint_of(egui::ViewportId::ROOT);
            settle();
            check(
                state.callbacks.load(SeqCst) > 1,
                "windowed root did not repaint",
            );
            // winit 0.30's macOS set_visible(true) itself makes the window key, independent of
            // eframe's explicit focus request. Test that older native limitation separately.
            #[cfg(target_os = "windows")]
            check(
                !state.focused.load(SeqCst),
                "inactive root stole focus after first presentation",
            );
            ctx.send_viewport_cmd_to(egui::ViewportId::ROOT, egui::ViewportCommand::Close);
            return failures;
        }

        // Start with zero windows, then create the first child on a timer-driven root pass.
        settle();
        state.show.store(true, SeqCst);
        ctx.request_repaint_after_for(Duration::from_millis(100), egui::ViewportId::ROOT);
        settle();
        let before = state.callbacks.load(SeqCst);
        check(before > 0, "timer did not create the child");
        ctx.request_repaint_of(child_id());
        settle();
        let after = state.callbacks.load(SeqCst);
        println!("Child repaint: {before} -> {after} callbacks");
        check(after > before, "child repaint did not wake its UI");
        check(after - before < 100, "child repaint caused an unpaced loop");

        // Removing the final native window must keep the logical controller usable.
        state.show.store(false, SeqCst);
        ctx.request_repaint_of(egui::ViewportId::ROOT);
        settle();
        let removed = state.callbacks.load(SeqCst);
        settle();
        check(
            state.callbacks.load(SeqCst) == removed,
            "removed child still repaints",
        );
        state.show.store(true, SeqCst);
        ctx.request_repaint_of(egui::ViewportId::ROOT);
        settle();
        check(
            state.callbacks.load(SeqCst) > removed,
            "child did not reopen",
        );

        state.cancel_next_close.store(true, SeqCst);
        ctx.send_viewport_cmd_to(egui::ViewportId::ROOT, egui::ViewportCommand::Close);
        settle();
        check(
            state.cancelled_closes.load(SeqCst) == 1,
            "root close was not cancelled",
        );
        // Disable cancellation before the final request, including on a failed check.
        state.cancel_next_close.store(false, SeqCst);
        ctx.send_viewport_cmd_to(egui::ViewportId::ROOT, egui::ViewportCommand::Close);
        failures
    }

    /// Reuses the native event loop for three child arrangements and a windowed root.
    /// Also checks first-presentation activation on Windows, where showing does not imply focus.
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let renderer = match std::env::args().nth(1).as_deref() {
            Some("glow") => eframe::Renderer::Glow,
            Some("wgpu") => eframe::Renderer::Wgpu,
            _ => return Err("pass glow or wgpu as the renderer".into()),
        };
        for mode in [
            Mode::Immediate,
            Mode::NestedImmediate,
            Mode::Deferred,
            Mode::Windowed,
        ] {
            println!("Checking {renderer:?}: {mode:?}");
            let mut worker = None;
            let result = eframe::run_native(
                "eframe windowless lifecycle check",
                eframe::NativeOptions {
                    renderer,
                    root_viewport_mode: if matches!(mode, Mode::Windowed) {
                        eframe::RootViewportMode::Windowed
                    } else {
                        eframe::RootViewportMode::Windowless
                    },
                    viewport: builder(),
                    persist_window: false,
                    ..Default::default()
                },
                Box::new(|cc| {
                    assert_eq!(
                        cc.winit_window().is_some(),
                        matches!(mode, Mode::Windowed),
                        "creation context must expose a window only in windowed mode"
                    );
                    let state = Arc::new(State::default());
                    let ctx = cc.egui_ctx.clone();
                    let control = Arc::clone(&state);
                    worker = Some(
                        thread::Builder::new()
                            .name("windowless lifecycle check".to_owned())
                            .spawn(move || exercise(&ctx, &control, mode))?,
                    );
                    Ok(Box::new(Probe { mode, state }))
                }),
            );
            let failures = worker.map(|worker| worker.join().expect("probe worker panicked"));
            result?;
            if let Some(failures) = failures
                && !failures.is_empty()
            {
                return Err(format!("{mode:?}: {}", failures.join("; ")).into());
            }
        }
        println!("All windowless lifecycle checks passed for {renderer:?}");
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    return native::run();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    Err("windowless roots currently require macOS or Windows".into())
}
