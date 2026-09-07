//! Completed native frame timings, including deferred and immediate viewports.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One completed native viewport frame, or one windowless controller update.
#[derive(Clone, Copy, Debug)]
pub struct ViewportFrameTiming {
    /// Viewport that performed the measured work. ROOT may be a windowless controller.
    pub viewport_id: egui::ViewportId,
    /// Monotonic completion time. Use the same clock when ageing a history of these samples.
    pub completed_at: Instant,
    /// Elapsed seconds spent preparing, running UI, tessellating, and submitting rendering.
    ///
    /// Excludes known presentation waits, nested immediate viewport work, and autosave.
    /// This is wall time on the UI thread, not process CPU utilization or GPU execution time.
    pub cpu_usage: f32,
}

type Callback = Arc<dyn Fn(ViewportFrameTiming) + Send + Sync>;

#[derive(Default)]
struct TimingState {
    callback: Option<Callback>,
    /// Full elapsed time of immediate children in each currently executing native frame.
    children: Vec<Duration>,
}

type SharedTiming = Arc<Mutex<TimingState>>;

fn timing_state(ctx: &egui::Context) -> SharedTiming {
    ctx.data_mut(|data| {
        data.get_temp_mut_or_default::<SharedTiming>(egui::Id::new("eframe_frame_timing"))
            .clone()
    })
}

/// Observes every completed native frame, including frames between root updates.
///
/// Install once during app creation. The callback runs synchronously after frame work, outside
/// egui context locks. It should only enqueue/store the sample, must not block or reenter the
/// renderer, and must not request an immediate repaint for every sample. Pass `None` to disable.
/// This API does not retain samples and does not change repaint scheduling.
pub fn set_viewport_frame_timing_callback(ctx: &egui::Context, callback: Option<Callback>) {
    timing_state(ctx).lock().unwrap().callback = callback;
}

/// Native frame scope. Drop excludes even abandoned child work from its parent's measurement.
pub(crate) struct ViewportFrameTimer {
    state: SharedTiming,
    viewport_id: egui::ViewportId,
    started: Instant,
    excluded: Duration,
    paused_at: Option<Instant>,
    finished: bool,
}

impl ViewportFrameTimer {
    pub fn new(ctx: &egui::Context, viewport_id: egui::ViewportId) -> Self {
        Self::new_at(timing_state(ctx), viewport_id, Instant::now())
    }

    fn new_at(state: SharedTiming, viewport_id: egui::ViewportId, started: Instant) -> Self {
        state.lock().unwrap().children.push(Duration::ZERO);
        Self {
            state,
            viewport_id,
            started,
            excluded: Duration::ZERO,
            paused_at: None,
            finished: false,
        }
    }

    #[cfg(feature = "glow")]
    pub fn pause(&mut self) {
        assert!(self.paused_at.is_none());
        self.paused_at = Some(Instant::now());
    }

    #[cfg(feature = "glow")]
    pub fn resume(&mut self) {
        self.excluded += self
            .paused_at
            .take()
            .expect("timer is not paused")
            .elapsed();
    }

    /// Subtracts the backend's measured acquisition/presentation wait for this frame only.
    #[cfg(any(feature = "wgpu_no_default_features", test))]
    pub fn exclude_seconds(&mut self, seconds: f32) {
        if seconds.is_finite() && seconds > 0.0 {
            self.excluded += Duration::from_secs_f32(seconds);
        }
    }

    /// Reports exactly one sample, after UI/output processing and before autosave.
    pub fn finish(mut self) -> f32 {
        self.finish_at(Instant::now(), true)
    }

    fn finish_at(&mut self, completed_at: Instant, report: bool) -> f32 {
        assert!(!self.finished);
        let elapsed = completed_at.saturating_duration_since(self.started);
        let (children, callback) = {
            let mut state = self.state.lock().unwrap();
            let children = state
                .children
                .pop()
                .expect("native timer scopes must be nested");
            if let Some(parent_children) = state.children.last_mut() {
                *parent_children += elapsed;
            }
            (children, state.callback.clone())
        };
        let paused = self.paused_at.map_or(Duration::ZERO, |at| {
            completed_at.saturating_duration_since(at)
        });
        let seconds = elapsed
            .saturating_sub(self.excluded + paused + children)
            .as_secs_f32();
        self.finished = true;
        if report && let Some(callback) = callback {
            callback(ViewportFrameTiming {
                viewport_id: self.viewport_id,
                completed_at,
                cpu_usage: seconds,
            });
        }
        seconds
    }
}

impl Drop for ViewportFrameTimer {
    fn drop(&mut self) {
        if !self.finished {
            self.finish_at(Instant::now(), false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::ViewportId;

    #[test]
    fn nested_frames_exclude_child_work_and_waits_without_losing_samples() {
        let state = SharedTiming::default();
        let samples = Arc::new(Mutex::new(Vec::new()));
        let sink = samples.clone();
        state.lock().unwrap().callback = Some(Arc::new(move |sample| {
            sink.lock().unwrap().push(sample);
        }));
        let at = Instant::now();
        let mut root = ViewportFrameTimer::new_at(state.clone(), ViewportId::ROOT, at);
        let child_id = ViewportId::from_hash_of("child");
        let mut child =
            ViewportFrameTimer::new_at(state.clone(), child_id, at + Duration::from_millis(2));
        let mut nested = ViewportFrameTimer::new_at(
            state.clone(),
            ViewportId::from_hash_of("nested"),
            at + Duration::from_millis(3),
        );
        nested.exclude_seconds(0.002);
        nested.finish_at(at + Duration::from_millis(8), true);
        child.exclude_seconds(0.004);
        child.finish_at(at + Duration::from_millis(14), true);
        root.finish_at(at + Duration::from_millis(20), true);
        let samples = samples.lock().unwrap();
        assert_eq!(samples.len(), 3);
        for (sample, expected) in samples.iter().zip([0.003, 0.003, 0.008]) {
            assert!((sample.cpu_usage - expected).abs() < 1e-6);
        }
        assert_eq!(samples[1].viewport_id, child_id);
        assert_eq!(samples[2].viewport_id, ViewportId::ROOT);
        assert!(state.lock().unwrap().children.is_empty());
    }

    #[test]
    fn failed_frame_does_not_publish_or_leak_its_scope() {
        let ctx = egui::Context::default();
        let samples = Arc::new(Mutex::new(Vec::new()));
        let sink = samples.clone();
        set_viewport_frame_timing_callback(
            &ctx,
            Some(Arc::new(move |sample| {
                sink.lock().unwrap().push(sample);
            })),
        );
        drop(ViewportFrameTimer::new(&ctx, egui::ViewportId::ROOT));
        assert!(samples.lock().unwrap().is_empty());
        assert!(timing_state(&ctx).lock().unwrap().children.is_empty());
        ViewportFrameTimer::new(&ctx, egui::ViewportId::ROOT).finish();
        assert_eq!(samples.lock().unwrap().len(), 1);
    }
}
