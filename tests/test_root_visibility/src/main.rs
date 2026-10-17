//! Self-terminating native regression probe for root visibility and deferred-window repainting.
//!
//! Run with `--renderer wgpu|glow` and `--initial hidden|default|visible`. The root starts
//! hidden until the first frame, follows the builder request, then responds to explicit
//! show/hide commands while a deferred child continues repainting. Add `--no-child` to exercise
//! hidden background logic without any visible windows. A watchdog bounds hangs.
#![allow(clippy::print_stdout)]

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use eframe::egui::{self, ViewportCommand, ViewportId};

const SETTLE: Duration = Duration::from_millis(250);
const REPAINT: Duration = Duration::from_millis(16);

/// Counts actual root/child callbacks, allowing the probe to detect a stalled hidden controller.
#[derive(Default)]
struct Observations {
    root_passes: AtomicUsize,
    logic_passes: AtomicUsize,
    child_passes: AtomicUsize,
    passed: AtomicBool,
    finished: AtomicBool,
}

/// Timed transitions are driven from App::logic, including while the root is occluded.
struct Probe {
    root: Arc<winit::window::Window>,
    observations: Arc<Observations>,
    initially_visible: bool,
    child_enabled: bool,
    // 0: initial, 1: shown, 2: hidden, 3: reshown, 4: rehidden, 5: closing.
    stage: usize,
    since: Instant,
    root_before_hide: usize,
    child_before_hide: usize,
}

impl eframe::App for Probe {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.observations
            .logic_passes
            .fetch_add(1, Ordering::SeqCst);
        ctx.request_repaint_after(REPAINT);
        // Observe every callback during hidden startup, not just the end of the settle period.
        if self.stage == 0 && !self.initially_visible {
            assert_eq!(
                self.root.is_visible(),
                Some(false),
                "hidden root was revealed"
            );
        }
        if self.since.elapsed() < SETTLE {
            return;
        }
        println!(
            "stage={} root_visible={:?} root_passes={} child_passes={} viewports={:?}",
            self.stage,
            self.root.is_visible(),
            self.observations.root_passes.load(Ordering::SeqCst),
            self.observations.child_passes.load(Ordering::SeqCst),
            ctx.input(|i| i
                .raw
                .viewports
                .iter()
                .map(|(id, info)| (*id, info.visible()))
                .collect::<Vec<_>>()),
        );
        let expected = match self.stage {
            0 => self.initially_visible,
            1 | 3 => true,
            2 | 4 => false,
            _ => return,
        };
        assert_eq!(
            self.root.is_visible(),
            Some(expected),
            "stage {}",
            self.stage
        );
        if self.stage == 0 {
            assert!(self.observations.root_passes.load(Ordering::SeqCst) > 0);
            if self.child_enabled {
                assert!(self.observations.child_passes.load(Ordering::SeqCst) > 0);
            }
        }
        if self.stage == 2 && self.child_enabled {
            assert!(
                self.observations.root_passes.load(Ordering::SeqCst) > self.root_before_hide,
                "hidden root UI stopped servicing its visible child"
            );
            assert!(
                self.observations.child_passes.load(Ordering::SeqCst) > self.child_before_hide,
                "deferred child stopped repainting while the root was hidden"
            );
        }
        if self.stage == 4 {
            self.observations.passed.store(true, Ordering::SeqCst);
            ctx.send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::Close);
        } else {
            let visible = self.stage % 2 == 0;
            ctx.send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::Visible(visible));
            if !visible {
                self.root_before_hide = self.observations.root_passes.load(Ordering::SeqCst);
                self.child_before_hide = self.observations.child_passes.load(Ordering::SeqCst);
            }
        }
        self.stage += 1;
        self.since = Instant::now();
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.observations.root_passes.fetch_add(1, Ordering::SeqCst);
        ui.label("Root visibility probe");
        if !self.child_enabled {
            return;
        }
        let observations = Arc::clone(&self.observations);
        ui.ctx().show_viewport_deferred(
            ViewportId::from_hash_of("visibility-probe-child"),
            egui::ViewportBuilder::default()
                .with_title("Root visibility probe - child")
                .with_position([500.0, 200.0])
                .with_active(false)
                .with_inner_size([320.0, 140.0]),
            move |ui, _| {
                let count = observations.child_passes.fetch_add(1, Ordering::SeqCst);
                ui.label(format!("Deferred frames: {count}"));
                ui.ctx().request_repaint_after(REPAINT);
                ui.ctx()
                    .request_repaint_after_for(REPAINT, ViewportId::ROOT);
            },
        );
    }
}

fn main() -> eframe::Result {
    let arguments: Vec<_> = std::env::args().collect();
    let option = |name| {
        arguments
            .windows(2)
            .find(|pair| pair[0] == name)
            .map(|pair| pair[1].as_str())
    };
    let renderer = match option("--renderer").unwrap_or("wgpu") {
        "wgpu" => eframe::Renderer::Wgpu,
        "glow" => eframe::Renderer::Glow,
        value => panic!("Unknown renderer: {value}"),
    };
    let initial = option("--initial").unwrap_or("hidden");
    let child_enabled = !arguments.iter().any(|arg| arg == "--no-child");
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("Root visibility probe - controller")
        .with_position([100.0, 200.0])
        .with_active(false)
        .with_inner_size([320.0, 140.0]);
    match initial {
        "hidden" => viewport = viewport.with_visible(false),
        "visible" => viewport = viewport.with_visible(true),
        "default" => {}
        value => panic!("Unknown initial visibility: {value}"),
    }
    let observations = Arc::new(Observations::default());
    let watchdog = Arc::clone(&observations);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(20));
        if !watchdog.finished.load(Ordering::SeqCst) {
            // A native event-loop deadlock must fail the probe rather than strand its windows.
            std::process::exit(2);
        }
    });
    let app_observations = Arc::clone(&observations);
    let result = eframe::run_native(
        "Root visibility probe",
        eframe::NativeOptions {
            renderer,
            viewport,
            ..Default::default()
        },
        Box::new(move |cc| {
            let root = Arc::clone(cc.winit_window().expect("native root window"));
            assert_eq!(root.is_visible(), Some(false), "shown before first paint");
            Ok(Box::new(Probe {
                root,
                observations: app_observations,
                initially_visible: initial != "hidden",
                child_enabled,
                stage: 0,
                since: Instant::now(),
                root_before_hide: 0,
                child_before_hide: 0,
            }))
        }),
    );
    observations.finished.store(true, Ordering::SeqCst);
    result?;
    assert!(
        observations.passed.load(Ordering::SeqCst),
        "probe exited early"
    );
    println!(
        concat!(
            "PASS renderer={renderer:?} initial={initial} child={child_enabled} ",
            "root_passes={} child_passes={} logic_passes={}",
        ),
        observations.root_passes.load(Ordering::SeqCst),
        observations.child_passes.load(Ordering::SeqCst),
        observations.logic_passes.load(Ordering::SeqCst),
        renderer = renderer,
        initial = initial,
        child_enabled = child_enabled,
    );
    Ok(())
}
