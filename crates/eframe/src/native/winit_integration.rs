use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use winit::{
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use egui::ViewportId;
#[cfg(feature = "accesskit")]
use egui_winit::accesskit_winit;

/// Returns `true` if the window is invisible or minimized.
///
/// These windows don't receive `RedrawRequested` events on Windows,
/// so they need special handling to keep processing viewport commands.
pub fn is_invisible_or_minimized(window: &Window) -> bool {
    window.is_visible() == Some(false) || window.is_minimized() == Some(true)
}

/// State accumulated while folding commands into a viewport that has no native window yet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PreCreationViewportState {
    /// The final unmatched close request, after applying all close cancellations in order.
    pub closing: bool,
    /// A focus action to defer until the hidden first frame has been presented and revealed.
    pub pending_focus: bool,
}

/// Applies creation-time viewport commands to the effective builder before a window exists.
///
/// Stateful commands are removed from `commands`; action commands retain their relative order.
/// Close/cancel pairs are reduced to at most one final `Close`, while focus is retained separately
/// so hidden-and-inactive-first creation cannot activate the application prematurely.
pub(crate) fn fold_pre_creation_viewport_commands(
    builder: &mut egui::ViewportBuilder,
    commands: &mut Vec<egui::ViewportCommand>,
) -> PreCreationViewportState {
    use egui::ViewportCommand;

    let mut state = PreCreationViewportState::default();
    let mut remaining = Vec::with_capacity(commands.len());
    for command in std::mem::take(commands) {
        if apply_stateful_viewport_command_to_builder(builder, &command) {
            continue;
        }
        match command {
            ViewportCommand::Close => state.closing = true,
            ViewportCommand::CancelClose => state.closing = false,
            ViewportCommand::Focus => state.pending_focus = true,
            command => remaining.push(command),
        }
    }

    if state.closing {
        // A close observation pass still needs the event, but the native window must never flash.
        builder.visible = Some(false);
        remaining.push(ViewportCommand::Close);
    }
    *commands = remaining;
    state
}

/// Effects produced while applying commands to a logical root without a native window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LogicalRootCommandState {
    /// Whether processing queued input requires another logical-root pass.
    pub repaint: bool,
    /// Whether the root icon changed and renderer-owned inheritance state must be synchronized.
    pub icon_changed: bool,
}

/// Applies viewport commands to a permanently windowless logical root.
///
/// Window-independent input commands become events for the next application pass. Title and icon
/// commands update the root's metadata so deferred children can continue inheriting it. Native
/// window commands have no target and are intentionally ignored.
pub(crate) fn process_logical_root_commands(
    builder: &mut egui::ViewportBuilder,
    info: &mut egui::ViewportInfo,
    input_events: &mut Vec<egui::Event>,
    platform: &mut egui_winit::WindowIndependentState,
    commands: Vec<egui::ViewportCommand>,
) -> LogicalRootCommandState {
    use egui::ViewportCommand;

    let mut state = LogicalRootCommandState::default();
    for command in commands {
        match command {
            ViewportCommand::Close => {
                info.events.push(egui::ViewportEvent::Close);
                state.repaint = true;
            }
            ViewportCommand::CancelClose => {}
            ViewportCommand::Title(title) => {
                info.title = Some(title.clone());
                builder.title = Some(title);
            }
            ViewportCommand::Icon(icon) => {
                builder.icon = icon;
                state.icon_changed = true;
            }
            ViewportCommand::RequestCut => {
                input_events.push(egui::Event::Cut);
                state.repaint = true;
            }
            ViewportCommand::RequestCopy => {
                input_events.push(egui::Event::Copy);
                state.repaint = true;
            }
            ViewportCommand::RequestPaste => {
                if let Some(text) = platform.clipboard_text() {
                    let text = text.replace("\r\n", "\n");
                    if !text.is_empty() {
                        input_events.push(egui::Event::Paste(text));
                        state.repaint = true;
                    }
                }
            }
            ViewportCommand::Screenshot(_) => {
                log::warn!("Screenshot is unavailable for a windowless root");
            }
            command => {
                log::debug!("Ignoring {command:?} for a windowless root");
            }
        }
    }
    state
}

/// Records a stateful command in the effective builder used for later native recreation.
pub(crate) fn apply_stateful_viewport_command_to_builder(
    builder: &mut egui::ViewportBuilder,
    command: &egui::ViewportCommand,
) -> bool {
    use egui::ViewportCommand;

    match command {
        ViewportCommand::Title(value) => builder.title = Some(value.clone()),
        ViewportCommand::Transparent(value) => builder.transparent = Some(*value),
        ViewportCommand::Visible(value) => builder.visible = Some(*value),
        ViewportCommand::OuterPosition(value) => builder.position = Some(*value),
        ViewportCommand::InnerSize(value) => builder.inner_size = Some(*value),
        ViewportCommand::MinInnerSize(value) => builder.min_inner_size = Some(*value),
        ViewportCommand::MaxInnerSize(value) => builder.max_inner_size = Some(*value),
        ViewportCommand::Resizable(value) => builder.resizable = Some(*value),
        ViewportCommand::EnableButtons {
            close,
            minimized,
            maximize,
        } => {
            builder.close_button = Some(*close);
            builder.minimize_button = Some(*minimized);
            builder.maximize_button = Some(*maximize);
        }
        ViewportCommand::Maximized(value) => builder.maximized = Some(*value),
        ViewportCommand::Fullscreen(value) => builder.fullscreen = Some(*value),
        ViewportCommand::SetMonitor(value) => builder.monitor = Some(*value),
        ViewportCommand::Decorations(value) => builder.decorations = Some(*value),
        ViewportCommand::WindowLevel(value) => builder.window_level = Some(*value),
        ViewportCommand::Icon(value) => builder.icon = value.clone(),
        ViewportCommand::MousePassthrough(value) => builder.mouse_passthrough = Some(*value),
        _ => return false,
    }
    true
}

/// Create an egui context, restoring it from storage if possible.
pub fn create_egui_context(storage: Option<&dyn crate::Storage>) -> egui::Context {
    profiling::function_scope!();

    pub const IS_DESKTOP: bool = cfg!(any(
        target_os = "freebsd",
        target_os = "linux",
        target_os = "macos",
        target_os = "openbsd",
        target_os = "windows",
    ));

    let egui_ctx = egui::Context::default();

    egui_ctx.set_embed_viewports(!IS_DESKTOP);

    egui_ctx.options_mut(|o| {
        // eframe supports multi-pass (Context::request_discard).
        #[expect(clippy::unwrap_used)]
        {
            o.max_passes = 2.try_into().unwrap();
        }
    });

    let memory = crate::native::epi_integration::load_egui_memory(storage).unwrap_or_default();
    egui_ctx.memory_mut(|mem| *mem = memory);

    egui_ctx
}

/// The custom even `eframe` uses with the [`winit`] event loop.
#[derive(Debug)]
pub enum UserEvent {
    /// A repaint is requested.
    RequestRepaint {
        /// What to repaint.
        viewport_id: ViewportId,

        /// When to repaint.
        when: Instant,

        /// Original deadline before egui compensated for predicted presentation time.
        requested_when: Instant,

        /// Whether the application genuinely requested an immediate repaint.
        genuinely_immediate: bool,

        /// What the cumulative pass number was when the repaint was _requested_.
        cumulative_pass_nr: u64,
    },

    /// A request related to [`accesskit`](https://accesskit.dev/).
    #[cfg(feature = "accesskit")]
    AccessKitActionRequest(accesskit_winit::Event),
}

#[cfg(feature = "accesskit")]
impl From<accesskit_winit::Event> for UserEvent {
    fn from(inner: accesskit_winit::Event) -> Self {
        Self::AccessKitActionRequest(inner)
    }
}

pub trait WinitApp {
    fn egui_ctx(&self) -> Option<&egui::Context>;

    fn window(&self, window_id: WindowId) -> Option<Arc<Window>>;

    fn window_id_from_viewport_id(&self, id: ViewportId) -> Option<WindowId>;

    fn save(&mut self);

    fn save_and_destroy(&mut self);

    fn run_ui_and_paint(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
    ) -> crate::Result<EventResult>;

    /// Whether this application owns a logical root with no native window.
    fn has_logical_root(&self) -> bool {
        false
    }

    /// Minimum interval between repeated immediate logical-root passes.
    fn logical_root_repaint_interval(&self) -> Duration {
        Duration::from_millis(100)
    }

    /// Run one logical-root application pass.
    fn run_logical_root(&mut self, _event_loop: &ActiveEventLoop) -> crate::Result<EventResult> {
        Ok(EventResult::Wait)
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) -> crate::Result<EventResult>;

    fn resumed(&mut self, event_loop: &ActiveEventLoop) -> crate::Result<EventResult>;

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) -> crate::Result<EventResult>;

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) -> crate::Result<EventResult>;

    #[cfg(feature = "accesskit")]
    fn on_accesskit_event(&mut self, event: accesskit_winit::Event) -> crate::Result<EventResult>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventResult {
    Wait,

    /// Causes a synchronous repaint inside the event handler. This should only
    /// be used in special situations if the window must be repainted while
    /// handling a specific event. This occurs on Windows when handling resizes.
    ///
    /// `RepaintNow` creates a new frame synchronously, and should therefore
    /// only be used for extremely urgent repaints.
    RepaintNow(WindowId),

    /// Queues a repaint for once the event loop handles its next redraw. Exists
    /// so that multiple input events can be handled in one frame. Does not
    /// cause any delay like `RepaintNow`.
    RepaintNext(WindowId),

    RepaintAt(WindowId, Instant),

    /// Run the windowless logical root at the next scheduler opportunity.
    RepaintLogicalRootNow,

    /// Run the windowless logical root at the given deadline.
    RepaintLogicalRootAt(Instant),

    /// Causes a save of the client state when the persistence feature is enabled.
    Save,

    /// Starts the process of ending eframe execution whilst allowing for proper
    /// clean up of resources.
    ///
    /// # Warning
    /// This event **must** occur before [`Exit`] to correctly exit eframe code.
    /// If in doubt, return this event.
    ///
    /// [`Exit`]: [EventResult::Exit]
    CloseRequested,

    /// Destroy application state and exit without waiting for another native-window event.
    CloseRequestedAndExit,

    /// The event loop will exit, now.
    /// The correct circumstance to return this event is in response to a winit "Destroyed" event.
    ///
    /// # Warning
    /// The [`CloseRequested`] **must** occur before this event to ensure that winit
    /// is able to remove any open windows. Otherwise the window(s) will remain open
    /// until the program terminates.
    ///
    /// [`CloseRequested`]: EventResult::CloseRequested
    Exit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_creation_commands_fold_into_effective_hidden_builder() {
        let mut builder = egui::ViewportBuilder::default();
        let mut commands = vec![
            egui::ViewportCommand::Title("child".to_owned()),
            egui::ViewportCommand::InnerSize(egui::vec2(713.0, 421.0)),
            egui::ViewportCommand::Visible(true),
            egui::ViewportCommand::Focus,
            egui::ViewportCommand::Close,
        ];

        let state = fold_pre_creation_viewport_commands(&mut builder, &mut commands);

        assert_eq!(builder.title.as_deref(), Some("child"));
        assert_eq!(builder.inner_size, Some(egui::vec2(713.0, 421.0)));
        assert_eq!(builder.visible, Some(false));
        assert!(state.closing);
        assert!(state.pending_focus);
        assert_eq!(commands, vec![egui::ViewportCommand::Close]);
    }

    #[test]
    fn pre_creation_cancel_close_avoids_terminal_close() {
        let mut builder = egui::ViewportBuilder::default();
        let mut commands = vec![
            egui::ViewportCommand::Close,
            egui::ViewportCommand::CancelClose,
            egui::ViewportCommand::Visible(false),
            egui::ViewportCommand::RequestCopy,
        ];

        let state = fold_pre_creation_viewport_commands(&mut builder, &mut commands);

        assert!(!state.closing);
        assert!(!state.pending_focus);
        assert_eq!(builder.visible, Some(false));
        assert_eq!(commands, vec![egui::ViewportCommand::RequestCopy]);
    }

    #[test]
    fn queued_close_reaches_a_windowless_logical_root() {
        let mut builder = egui::ViewportBuilder::default();
        let mut commands = vec![egui::ViewportCommand::Close];
        let state = fold_pre_creation_viewport_commands(&mut builder, &mut commands);
        assert!(state.closing);

        let mut info = egui::ViewportInfo::default();
        let mut input_events = Vec::new();
        let mut platform = egui_winit::WindowIndependentState::new(None);
        let state = process_logical_root_commands(
            &mut builder,
            &mut info,
            &mut input_events,
            &mut platform,
            commands,
        );

        assert!(state.repaint);
        assert!(info.close_requested());
    }
}
