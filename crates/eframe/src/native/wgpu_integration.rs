//! Note that this file contains code very similar to [`super::glow_integration`].
//! When making changes to one you often also want to apply it to the other.
//!
//! This is also very complex code, and not very pretty.
//! There is a bunch of improvements we could do,
//! like removing a bunch of `unwraps`.

use std::{cell::RefCell, cmp::Reverse, num::NonZeroU32, rc::Rc, sync::Arc, time::Instant};

use egui_winit::ActionRequested;
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
use winit::{
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

use ahash::HashMap;
use egui::{
    DeferredViewportUiCallback, FullOutput, ImmediateViewport, OrderedViewportIdMap,
    ViewportBuilder, ViewportClass, ViewportId, ViewportIdPair, ViewportIdSet, ViewportInfo,
    ViewportOutput,
};
#[cfg(feature = "accesskit")]
use egui_winit::accesskit_winit;
use winit_integration::UserEvent;

use crate::{
    App, AppCreator, CreationContext, NativeOptions, Result, Storage,
    native::{
        epi_integration::EpiIntegration,
        winit_integration::{EventResult, is_invisible_or_minimized},
    },
};

use super::{epi_integration, event_loop_context, winit_integration, winit_integration::WinitApp};

// ----------------------------------------------------------------------------
// Types:

pub struct WgpuWinitApp<'app> {
    repaint_proxy: Arc<Mutex<EventLoopProxy<UserEvent>>>,
    app_name: String,
    native_options: NativeOptions,

    /// Set at initialization, then taken and set to `None` in `init_run_state`.
    app_creator: Option<AppCreator<'app>>,

    /// Set when we are actually up and running.
    running: Option<WgpuWinitRunning<'app>>,

    /// An optional pre-existing egui context. If `Some`, it is used instead of
    /// creating a new one via [`winit_integration::create_egui_context`]. Taken during initialization.
    egui_ctx: Option<egui::Context>,
}

/// State that is initialized when the application is first starts running via
/// a Resumed event. On Android this ensures that any graphics state is only
/// initialized once the application has an associated `SurfaceView`.
struct WgpuWinitRunning<'app> {
    integration: EpiIntegration,

    /// True when `ViewportId::ROOT` is application state only and has no native window.
    logical_root: bool,

    /// The users application.
    app: Box<dyn 'app + App>,

    /// Wrapped in an `Rc<RefCell<…>>` so it can be re-entrantly shared via a weak-pointer.
    shared: Rc<RefCell<SharedState>>,
}

/// Everything needed by the immediate viewport renderer.\
///
/// This is shared by all viewports.
///
/// Wrapped in an `Rc<RefCell<…>>` so it can be re-entrantly shared via a weak-pointer.
pub struct SharedState {
    egui_ctx: egui::Context,
    viewports: Viewports,
    painter: egui_wgpu::winit::Painter,
    viewport_from_window: HashMap<WindowId, ViewportId>,
    focused_viewport: Option<ViewportId>,
    root_platform: egui_winit::WindowIndependentState,
    root_events: Vec<egui::Event>,
    #[cfg(feature = "accesskit")]
    active_accesskit_windows: ahash::HashSet<WindowId>,
    next_focus_ordinal: u64,

    /// First backend failure raised inside an immediate viewport callback.
    ///
    /// The callback API is infallible, so the enclosing eframe dispatch observes and returns this
    /// error after the callback contract has been satisfied exactly once.
    fatal_error: Option<crate::Error>,

    /// `Some(NativeResize { state: NativeResizeState::Idle, .. })` must never be stored;
    /// an idle renderer is represented by `None`.
    native_resize: Option<NativeResize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NativeResize {
    viewport_id: ViewportId,
    state: egui_wgpu::winit::NativeResizeState,
}

impl SharedState {
    /// Applies commands addressed to the permanently windowless root.
    ///
    /// Commands can arrive while a child callback is running, so they remain queued until the
    /// controller's next pass. The return value requests that pass when new root input was added.
    fn process_logical_root_commands(&mut self, commands: Vec<egui::ViewportCommand>) -> bool {
        let root = self
            .viewports
            .get_mut(&ViewportId::ROOT)
            .expect("logical root viewport must exist");
        let state = winit_integration::process_logical_root_commands(
            &mut root.builder,
            &mut root.info,
            &mut self.root_events,
            &mut self.root_platform,
            commands,
        );
        if state.icon_changed {
            root.icon_state = ViewportIconState::Explicit(root.builder.icon.clone());
        }
        state.repaint
    }

    /// Switches the viewport whose Metal surface is configured for `AppKit` native resize.
    fn set_native_resize(&mut self, next: Option<NativeResize>) {
        debug_assert!(
            next.is_none_or(|resize| { resize.state != egui_wgpu::winit::NativeResizeState::Idle }),
            "idle native resize must be represented by None",
        );

        if self.native_resize == next {
            return;
        }

        if let Some(previous) = self.native_resize
            && next.map(|resize| resize.viewport_id) != Some(previous.viewport_id)
        {
            self.painter.set_native_resize_state(
                previous.viewport_id,
                egui_wgpu::winit::NativeResizeState::Idle,
            );
        }

        if let Some(next) = next {
            self.painter
                .set_native_resize_state(next.viewport_id, next.state);
        }

        self.native_resize = next;
    }

    /// Finalizes renderer resize state once `AppKit` reports that native resizing has ended.
    #[cfg(target_os = "macos")]
    fn finish_ended_native_resize(&mut self) {
        let Some(native_resize) = self.native_resize else {
            return;
        };

        let native_resize_state = self
            .viewports
            .get(&native_resize.viewport_id)
            .and_then(|viewport| viewport.window.as_deref())
            .map_or(
                egui_wgpu::winit::NativeResizeState::Idle,
                native_resize_state,
            );

        if native_resize_state == egui_wgpu::winit::NativeResizeState::Idle {
            self.set_native_resize(None);
        }
    }
}

#[cfg(target_os = "macos")]
fn native_resize_state(window: &Window) -> egui_wgpu::winit::NativeResizeState {
    use winit::platform::macos::WindowExtMacOS as _;

    if window.is_live_resizing() {
        egui_wgpu::winit::NativeResizeState::LiveResize
    } else if window.is_fullscreen_transition() {
        egui_wgpu::winit::NativeResizeState::FullscreenTransition
    } else {
        egui_wgpu::winit::NativeResizeState::Idle
    }
}

pub type Viewports = egui::OrderedViewportIdMap<Viewport>;

#[derive(Clone)]
enum ViewportIconState {
    Inherited,
    Explicit(Option<Arc<egui::IconData>>),
}

pub struct Viewport {
    ids: ViewportIdPair,
    declaration_ordinal: u64,
    class: ViewportClass,
    builder: ViewportBuilder,
    icon_state: ViewportIconState,
    deferred_commands: Vec<egui::viewport::ViewportCommand>,
    info: ViewportInfo,
    actions_requested: Vec<ActionRequested>,

    /// `None` for sync viewports.
    viewport_ui_cb: Option<Arc<DeferredViewportUiCallback>>,

    /// Window surface state that's initialized when the app starts running via a Resumed event
    /// and on Android will also be destroyed if the application is paused.
    window: Option<Arc<Window>>,

    /// `window` and `egui_winit` are initialized together.
    egui_winit: Option<egui_winit::State>,

    /// Visibility requested by the builder, applied only after the first successful frame.
    requested_visible: bool,
    /// Activation requested by the builder, applied only after the first successful frame.
    requested_active: bool,
    /// Focus action retained until a requested-visible child has presented its first frame.
    pending_focus: bool,
    has_presented: bool,
    currently_focused: bool,
    focus_ordinal: Option<u64>,
}

// ----------------------------------------------------------------------------

impl<'app> WgpuWinitApp<'app> {
    pub fn new(
        event_loop: &EventLoop<UserEvent>,
        app_name: &str,
        native_options: NativeOptions,
        egui_ctx: Option<egui::Context>,
        app_creator: AppCreator<'app>,
    ) -> Self {
        profiling::function_scope!();

        #[cfg(feature = "__screenshot")]
        assert!(
            std::env::var("EFRAME_SCREENSHOT_TO").is_err(),
            "EFRAME_SCREENSHOT_TO not yet implemented for wgpu backend"
        );

        Self {
            repaint_proxy: Arc::new(Mutex::new(event_loop.create_proxy())),
            app_name: app_name.to_owned(),
            native_options,
            running: None,
            app_creator: Some(app_creator),
            egui_ctx,
        }
    }

    /// Create a window for all viewports lacking one.
    fn initialized_all_windows(&mut self, event_loop: &ActiveEventLoop) -> crate::Result<()> {
        let Some(running) = &mut self.running else {
            return Ok(());
        };
        let mut shared = running.shared.borrow_mut();
        let SharedState {
            viewports,
            painter,
            viewport_from_window,
            ..
        } = &mut *shared;

        for viewport in viewports.values_mut() {
            if running.logical_root && viewport.ids.this == ViewportId::ROOT {
                continue;
            }
            let was_uninitialized = viewport.window.is_none();
            viewport.initialize_window(
                event_loop,
                &running.integration.egui_ctx,
                viewport_from_window,
                painter,
            )?;
            if was_uninitialized && let Some(window) = &viewport.window {
                window.request_redraw();
            }
        }
        Ok(())
    }

    #[cfg(target_os = "android")]
    fn recreate_window(
        &self,
        event_loop: &ActiveEventLoop,
        running: &WgpuWinitRunning<'app>,
    ) -> crate::Result<()> {
        let SharedState {
            egui_ctx,
            viewports,
            viewport_from_window,
            painter,
            ..
        } = &mut *running.shared.borrow_mut();

        initialize_or_update_viewport(
            viewports,
            ViewportIdPair::ROOT,
            0,
            ViewportClass::Root,
            self.native_options.viewport.clone(),
            None,
            painter,
        )
        .initialize_window(event_loop, egui_ctx, viewport_from_window, painter)?;
        Ok(())
    }

    #[cfg(target_os = "android")]
    fn drop_window(&mut self) -> Result<(), egui_wgpu::WgpuError> {
        if let Some(running) = &mut self.running {
            let mut shared = running.shared.borrow_mut();
            shared.viewports.remove(&ViewportId::ROOT);
            pollster::block_on(shared.painter.set_window(ViewportId::ROOT, None))?;
        }
        Ok(())
    }

    fn init_run_state(
        &mut self,
        egui_ctx: egui::Context,
        event_loop: &ActiveEventLoop,
        storage: Option<Box<dyn Storage>>,
        window: Window,
        builder: ViewportBuilder,
    ) -> crate::Result<&mut WgpuWinitRunning<'app>> {
        profiling::function_scope!();
        // Inject the display handle into the wgpu setup so that wgpu can create
        // surfaces on platforms that require it (e.g. GLES on Wayland).
        let mut wgpu_options = self.native_options.wgpu_options.clone();
        if let egui_wgpu::WgpuSetup::CreateNew(ref mut create_new) = wgpu_options.wgpu_setup
            && create_new.display_handle.is_none()
        {
            create_new.display_handle = Some(Box::new(event_loop.owned_display_handle()));
        }
        let mut painter = pollster::block_on(egui_wgpu::winit::Painter::new(
            egui_ctx.clone(),
            wgpu_options,
            self.native_options.viewport.transparent.unwrap_or(false),
            egui_wgpu::RendererOptions {
                msaa_samples: self.native_options.multisampling as _,
                depth_stencil_format: egui_wgpu::depth_format_from_bits(
                    self.native_options.depth_buffer,
                    self.native_options.stencil_buffer,
                ),
                dithering: self.native_options.dithering,
                ..Default::default()
            },
        ));

        let mut viewport_info = ViewportInfo::default();
        egui_winit::update_viewport_info(&mut viewport_info, &egui_ctx, &window, true);

        {
            // Tell egui right away about native_pixels_per_point etc,
            // so that the app knows about it during app creation:
            let pixels_per_point = egui_winit::pixels_per_point(&egui_ctx, &window);

            egui_ctx.input_mut(|i| {
                i.raw
                    .viewports
                    .insert(ViewportId::ROOT, viewport_info.clone());
                i.pixels_per_point = pixels_per_point;
            });
        }

        let window = Arc::new(window);

        {
            profiling::scope!("set_window");
            pollster::block_on(painter.set_window(ViewportId::ROOT, Some(Arc::clone(&window))))?;
        }

        let wgpu_render_state = painter.render_state();

        let integration = EpiIntegration::new(
            egui_ctx.clone(),
            Some(&window),
            window.display_handle().map(|handle| handle.as_raw()),
            &self.app_name,
            &self.native_options,
            storage,
            #[cfg(feature = "glow")]
            None,
            #[cfg(feature = "glow")]
            None,
            wgpu_render_state.clone(),
        );

        {
            let event_loop_proxy = Arc::clone(&self.repaint_proxy);

            egui_ctx.set_request_repaint_callback(move |info| {
                log::trace!("request_repaint_callback: {info:?}");
                let now = Instant::now();
                let when = now + info.delay;
                let requested_when = now + info.requested_delay;
                let cumulative_pass_nr = info.current_cumulative_pass_nr;

                event_loop_proxy
                    .lock()
                    .send_event(UserEvent::RequestRepaint {
                        when,
                        requested_when,
                        genuinely_immediate: info.requested_delay.is_zero(),
                        cumulative_pass_nr,
                        viewport_id: info.viewport_id,
                    })
                    .ok();
            });
        }

        #[allow(clippy::allow_attributes, unused_mut)] // used for accesskit
        let mut egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            event_loop.system_theme(),
            painter.max_texture_side(),
        );

        #[cfg(feature = "accesskit")]
        {
            let event_loop_proxy = self.repaint_proxy.lock().clone();
            egui_winit.init_accesskit(event_loop, &window, event_loop_proxy);
        }

        let app_creator = std::mem::take(&mut self.app_creator)
            .expect("Single-use AppCreator has unexpectedly already been taken");

        crate::maybe_attach_inspection_plugin(&egui_ctx, Some(self.app_name.clone()));

        let cc = CreationContext {
            egui_ctx: egui_ctx.clone(),
            integration_info: integration.frame.info().clone(),
            storage: integration.frame.storage(),
            #[cfg(feature = "glow")]
            gl: None,
            #[cfg(feature = "glow")]
            get_proc_address: None,
            wgpu_render_state,
            window: Some(Arc::clone(&window)),
            raw_display_handle: window.display_handle().map(|h| h.as_raw()),
            raw_window_handle: window.window_handle().map(|h| h.as_raw()),
        };
        let app = {
            profiling::scope!("user_app_creator");
            app_creator(&cc).map_err(crate::Error::AppCreation)?
        };

        let mut viewport_from_window = HashMap::default();
        viewport_from_window.insert(window.id(), ViewportId::ROOT);

        let mut viewports = Viewports::default();
        viewports.insert(
            ViewportId::ROOT,
            Viewport {
                ids: ViewportIdPair::ROOT,
                declaration_ordinal: 0,
                class: ViewportClass::Root,
                builder,
                icon_state: ViewportIconState::Explicit(self.native_options.viewport.icon.clone()),
                deferred_commands: vec![],
                info: viewport_info,
                actions_requested: Default::default(),
                viewport_ui_cb: None,
                window: Some(window),
                egui_winit: Some(egui_winit),
                requested_visible: true,
                requested_active: true,
                pending_focus: false,
                has_presented: false,
                currently_focused: true,
                focus_ordinal: Some(0),
            },
        );

        let shared = Rc::new(RefCell::new(SharedState {
            egui_ctx,
            viewport_from_window,
            viewports,
            painter,
            focused_viewport: Some(ViewportId::ROOT),
            root_platform: egui_winit::WindowIndependentState::new(
                event_loop
                    .display_handle()
                    .ok()
                    .map(|handle| handle.as_raw()),
            ),
            root_events: vec![],
            #[cfg(feature = "accesskit")]
            active_accesskit_windows: Default::default(),
            next_focus_ordinal: 1,
            fatal_error: None,
            native_resize: None,
        }));

        {
            // Create a weak pointer so that we don't keep state alive for too long.
            let shared = Rc::downgrade(&shared);
            let beginning = integration.beginning;

            egui::Context::set_immediate_viewport_renderer(move |_egui_ctx, immediate_viewport| {
                if let Some(shared) = shared.upgrade() {
                    render_immediate_viewport(beginning, &shared, immediate_viewport);
                } else {
                    log::warn!("render_sync_callback called after window closed");
                }
            });
        }

        Ok(self.running.insert(WgpuWinitRunning {
            integration,
            app,
            shared,
            logical_root: false,
        }))
    }

    /// Initializes WGPU from a private surface and then starts the app with a logical root.
    ///
    /// The bootstrap window is hidden and inactive and is destroyed before user app creation.
    fn init_windowless_run_state(
        &mut self,
        egui_ctx: egui::Context,
        event_loop: &ActiveEventLoop,
        storage: Option<Box<dyn Storage>>,
    ) -> crate::Result<&mut WgpuWinitRunning<'app>> {
        profiling::function_scope!();

        epi_integration::ensure_default_egui_icon(&mut self.native_options.viewport);

        let mut wgpu_options = self.native_options.wgpu_options.clone();
        if let egui_wgpu::WgpuSetup::CreateNew(ref mut create_new) = wgpu_options.wgpu_setup
            && create_new.display_handle.is_none()
        {
            create_new.display_handle = Some(Box::new(event_loop.owned_display_handle()));
        }

        let mut painter = pollster::block_on(egui_wgpu::winit::Painter::new(
            egui_ctx.clone(),
            wgpu_options,
            false,
            egui_wgpu::RendererOptions {
                msaa_samples: self.native_options.multisampling as _,
                depth_stencil_format: egui_wgpu::depth_format_from_bits(
                    self.native_options.depth_buffer,
                    self.native_options.stencil_buffer,
                ),
                dithering: self.native_options.dithering,
                ..Default::default()
            },
        ));

        if self.native_options.window_builder.take().is_some() {
            log::warn!("`NativeOptions::window_builder` is ignored for a windowless root");
        }

        let bootstrap_builder = ViewportBuilder::default()
            .with_title("eframe renderer bootstrap")
            .with_inner_size([1.0, 1.0])
            .with_visible(false)
            .with_active(false)
            .with_decorations(false)
            .with_taskbar(false);
        let bootstrap_window = Arc::new(egui_winit::create_window(
            &egui_ctx,
            event_loop,
            &bootstrap_builder,
        )?);
        exclude_renderer_bootstrap_from_window_menu(&bootstrap_window);
        pollster::block_on(painter.set_window_with_transparency(
            ViewportId::ROOT,
            Some(Arc::clone(&bootstrap_window)),
            false,
        ))?;
        let wgpu_render_state = painter.render_state();
        painter.remove_surface(ViewportId::ROOT);
        drop(bootstrap_window);

        let root_builder = self.native_options.viewport.clone();
        let root_title = root_builder
            .title
            .clone()
            .unwrap_or_else(|| self.app_name.clone());
        let root_info = ViewportInfo {
            title: Some(root_title),
            focused: Some(false),
            ..Default::default()
        };
        let max_texture_side = painter.max_texture_side();
        let system_theme = event_loop.system_theme().map(|theme| match theme {
            winit::window::Theme::Light => egui::Theme::Light,
            winit::window::Theme::Dark => egui::Theme::Dark,
        });
        egui_ctx.input_mut(|input| {
            input.raw.focused = false;
            input.raw.system_theme = system_theme;
            input.raw.max_texture_side = max_texture_side;
            input
                .raw
                .viewports
                .insert(ViewportId::ROOT, root_info.clone());
        });

        let raw_display_handle = event_loop.display_handle().map(|handle| handle.as_raw());
        let integration = EpiIntegration::new(
            egui_ctx.clone(),
            None,
            raw_display_handle.clone(),
            &self.app_name,
            &self.native_options,
            storage,
            #[cfg(feature = "glow")]
            None,
            #[cfg(feature = "glow")]
            None,
            wgpu_render_state.clone(),
        );

        {
            let event_loop_proxy = Arc::clone(&self.repaint_proxy);
            egui_ctx.set_request_repaint_callback(move |info| {
                let now = Instant::now();
                event_loop_proxy
                    .lock()
                    .send_event(UserEvent::RequestRepaint {
                        when: now + info.delay,
                        requested_when: now + info.requested_delay,
                        genuinely_immediate: info.requested_delay.is_zero(),
                        cumulative_pass_nr: info.current_cumulative_pass_nr,
                        viewport_id: info.viewport_id,
                    })
                    .ok();
            });
        }

        let app_creator = std::mem::take(&mut self.app_creator)
            .expect("Single-use AppCreator has unexpectedly already been taken");
        crate::maybe_attach_inspection_plugin(&egui_ctx, Some(self.app_name.clone()));
        let cc = CreationContext {
            egui_ctx: egui_ctx.clone(),
            integration_info: integration.frame.info().clone(),
            storage: integration.frame.storage(),
            #[cfg(feature = "glow")]
            gl: None,
            #[cfg(feature = "glow")]
            get_proc_address: None,
            wgpu_render_state,
            window: None,
            raw_display_handle,
            raw_window_handle: Err(raw_window_handle::HandleError::NotSupported),
        };
        let app = app_creator(&cc).map_err(crate::Error::AppCreation)?;

        let mut viewports = Viewports::default();
        viewports.insert(
            ViewportId::ROOT,
            Viewport {
                ids: ViewportIdPair::ROOT,
                declaration_ordinal: 0,
                class: ViewportClass::Root,
                builder: root_builder,
                icon_state: ViewportIconState::Explicit(self.native_options.viewport.icon.clone()),
                deferred_commands: vec![],
                info: root_info,
                actions_requested: vec![],
                viewport_ui_cb: None,
                window: None,
                egui_winit: None,
                requested_visible: false,
                requested_active: false,
                pending_focus: false,
                has_presented: false,
                currently_focused: false,
                focus_ordinal: None,
            },
        );
        let shared = Rc::new(RefCell::new(SharedState {
            egui_ctx,
            viewports,
            painter,
            viewport_from_window: HashMap::default(),
            focused_viewport: None,
            root_platform: egui_winit::WindowIndependentState::new(
                event_loop
                    .display_handle()
                    .ok()
                    .map(|handle| handle.as_raw()),
            ),
            root_events: vec![],
            #[cfg(feature = "accesskit")]
            active_accesskit_windows: Default::default(),
            next_focus_ordinal: 1,
            fatal_error: None,
            native_resize: None,
        }));

        {
            let shared = Rc::downgrade(&shared);
            let beginning = integration.beginning;
            egui::Context::set_immediate_viewport_renderer(move |_ctx, viewport| {
                if let Some(shared) = shared.upgrade() {
                    render_immediate_viewport(beginning, &shared, viewport);
                }
            });
        }

        Ok(self.running.insert(WgpuWinitRunning {
            integration,
            app,
            shared,
            logical_root: true,
        }))
    }
}

/// Keeps the private Metal bootstrap out of the macOS Window menu during slow initialization.
#[cfg(target_os = "macos")]
#[expect(unsafe_code)]
fn exclude_renderer_bootstrap_from_window_menu(window: &Window) {
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let raw_window_handle::RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return;
    };
    let view = unsafe { &*handle.ns_view.as_ptr().cast::<objc2_app_kit::NSView>() };
    if let Some(window) = view.window() {
        window.setExcludedFromWindowsMenu(true);
    }
}

#[cfg(not(target_os = "macos"))]
fn exclude_renderer_bootstrap_from_window_menu(_window: &Window) {}

impl WinitApp for WgpuWinitApp<'_> {
    fn egui_ctx(&self) -> Option<&egui::Context> {
        self.running.as_ref().map(|r| &r.integration.egui_ctx)
    }

    fn window(&self, window_id: WindowId) -> Option<Arc<Window>> {
        self.running
            .as_ref()
            .and_then(|r| {
                let shared = r.shared.borrow();
                let id = shared.viewport_from_window.get(&window_id)?;
                shared.viewports.get(id).map(|v| v.window.clone())
            })
            .flatten()
    }

    fn window_id_from_viewport_id(&self, id: ViewportId) -> Option<WindowId> {
        Some(
            self.running
                .as_ref()?
                .shared
                .borrow()
                .viewports
                .get(&id)?
                .window
                .as_ref()?
                .id(),
        )
    }

    fn save(&mut self) {
        log::debug!("WinitApp::save called");
        if let Some(running) = self.running.as_mut() {
            running.save();
        }
    }

    fn save_and_destroy(&mut self) {
        if let Some(mut running) = self.running.take() {
            running.save_and_destroy();
        }
    }

    fn run_ui_and_paint(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
    ) -> Result<EventResult> {
        self.initialized_all_windows(event_loop)?;

        if let Some(running) = &mut self.running {
            running.run_ui_and_paint(window_id, event_loop)
        } else {
            Ok(EventResult::Wait)
        }
    }

    fn has_logical_root(&self) -> bool {
        self.running
            .as_ref()
            .is_some_and(|running| running.logical_root)
    }

    fn logical_root_repaint_interval(&self) -> std::time::Duration {
        self.running
            .as_ref()
            .map_or(std::time::Duration::from_millis(100), |running| {
                logical_root_repaint_interval(&running.shared.borrow().viewports)
            })
    }

    fn run_logical_root(&mut self, event_loop: &ActiveEventLoop) -> crate::Result<EventResult> {
        let Some(running) = &mut self.running else {
            return Ok(EventResult::Wait);
        };
        running.run_logical_root(event_loop)
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) -> crate::Result<EventResult> {
        log::debug!("Event::Resumed");

        let running = if let Some(running) = &self.running {
            #[cfg(target_os = "android")]
            self.recreate_window(event_loop, running)?;
            running
        } else {
            let storage = if let Some(file) = &self.native_options.persistence_path {
                epi_integration::create_storage_with_file(file)
            } else {
                epi_integration::create_storage(
                    self.native_options
                        .viewport
                        .app_id
                        .as_ref()
                        .unwrap_or(&self.app_name),
                )
            };
            let egui_ctx = self
                .egui_ctx
                .take()
                .unwrap_or_else(|| winit_integration::create_egui_context(storage.as_deref()));
            match self.native_options.root_viewport_mode {
                crate::RootViewportMode::Windowed => {
                    let (window, builder) = create_window(
                        &egui_ctx,
                        event_loop,
                        storage.as_deref(),
                        &mut self.native_options,
                    )?;
                    self.init_run_state(egui_ctx, event_loop, storage, window, builder)?
                }
                crate::RootViewportMode::Windowless => {
                    if !cfg!(any(target_os = "macos", target_os = "windows")) {
                        return Err(crate::Error::UnsupportedConfiguration(
                            "windowless root requires macOS or Windows".to_owned(),
                        ));
                    }
                    self.init_windowless_run_state(egui_ctx, event_loop, storage)?
                }
            }
        };

        let viewport = &running.shared.borrow().viewports[&ViewportId::ROOT];
        if running.logical_root {
            Ok(EventResult::RepaintLogicalRootNow)
        } else if let Some(window) = &viewport.window {
            Ok(EventResult::RepaintNow(window.id()))
        } else {
            Ok(EventResult::Wait)
        }
    }

    fn suspended(&mut self, _: &ActiveEventLoop) -> crate::Result<EventResult> {
        #[cfg(target_os = "android")]
        self.drop_window()?;
        Ok(EventResult::Save)
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) -> crate::Result<EventResult> {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event
            && let Some(running) = &mut self.running
        {
            let mut shared = running.shared.borrow_mut();
            if let Some(viewport) = shared
                .focused_viewport
                .and_then(|viewport| shared.viewports.get_mut(&viewport))
                && let Some(window) = viewport.window.as_ref()
            {
                if !window.has_focus()
                    && !viewport
                        .egui_winit
                        .as_ref()
                        .map(|state| state.is_any_pointer_button_down())
                        .unwrap_or(false)
                {
                    return Ok(EventResult::Wait);
                }

                if let Some(egui_winit) = viewport.egui_winit.as_mut()
                    && egui_winit.on_mouse_motion(delta)
                {
                    return Ok(EventResult::RepaintNext(window.id()));
                }
            }
        }

        Ok(EventResult::Wait)
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) -> crate::Result<EventResult> {
        self.initialized_all_windows(event_loop)?;

        if let Some(running) = &mut self.running {
            Ok(running.on_window_event(window_id, &event))
        } else {
            // running is removed to get ready for exiting
            Ok(EventResult::Exit)
        }
    }

    #[cfg(feature = "accesskit")]
    fn on_accesskit_event(&mut self, event: accesskit_winit::Event) -> crate::Result<EventResult> {
        if let Some(running) = &mut self.running {
            let mut shared_lock = running.shared.borrow_mut();
            let viewport_id = shared_lock
                .viewport_from_window
                .get(&event.window_id)
                .copied();
            match &event.window_event {
                accesskit_winit::WindowEvent::InitialTreeRequested => {
                    let was_empty = shared_lock.active_accesskit_windows.is_empty();
                    shared_lock.active_accesskit_windows.insert(event.window_id);
                    if was_empty {
                        running.integration.egui_ctx.enable_accesskit();
                    }
                    return Ok(EventResult::RepaintNow(event.window_id));
                }
                accesskit_winit::WindowEvent::AccessibilityDeactivated => {
                    shared_lock
                        .active_accesskit_windows
                        .remove(&event.window_id);
                    if shared_lock.active_accesskit_windows.is_empty() {
                        running.integration.egui_ctx.disable_accesskit();
                    }
                    return Ok(EventResult::Wait);
                }
                accesskit_winit::WindowEvent::ActionRequested(request) => {
                    if let Some(viewport) =
                        viewport_id.and_then(|id| shared_lock.viewports.get_mut(&id))
                        && let Some(egui_winit) = &mut viewport.egui_winit
                    {
                        egui_winit.on_accesskit_action_request(request.clone());
                        return Ok(EventResult::RepaintNext(event.window_id));
                    }
                }
            }
        }

        Ok(EventResult::Wait)
    }
}

impl WgpuWinitRunning<'_> {
    /// Saves the application state
    fn save(&mut self) {
        let shared = self.shared.borrow();
        // This is done because of the "save on suspend" logic on Android. Once the application is suspended, there is no window associated to it.
        let window = if let Some(Viewport { window, .. }) = shared.viewports.get(&ViewportId::ROOT)
        {
            window.as_deref()
        } else {
            None
        };
        self.integration.save(self.app.as_mut(), window);
    }

    fn save_and_destroy(&mut self) {
        profiling::function_scope!();

        self.save();

        #[cfg(feature = "glow")]
        self.app.on_exit(None);

        #[cfg(not(feature = "glow"))]
        self.app.on_exit();

        let mut shared = self.shared.borrow_mut();
        shared.painter.destroy();
    }

    /// Runs one application pass for the root viewport without a native window or surface.
    fn run_logical_root(&mut self, event_loop: &ActiveEventLoop) -> Result<EventResult> {
        profiling::function_scope!();

        let raw_input = {
            let mut shared = self.shared.borrow_mut();
            let commands = shared
                .viewports
                .get_mut(&ViewportId::ROOT)
                .map(|root| std::mem::take(&mut root.deferred_commands))
                .unwrap_or_default();
            let _ = shared.process_logical_root_commands(commands);
            let root_events = std::mem::take(&mut shared.root_events);
            let viewports = shared
                .viewports
                .iter()
                .map(|(id, viewport)| (*id, viewport.info.clone()))
                .collect();
            let predicted_dt = logical_root_repaint_interval(&shared.viewports).as_secs_f32();

            egui::RawInput {
                viewport_id: ViewportId::ROOT,
                viewports,
                max_texture_side: shared.painter.max_texture_side(),
                predicted_dt,
                focused: false,
                system_theme: event_loop.system_theme().map(|theme| match theme {
                    winit::window::Theme::Light => egui::Theme::Light,
                    winit::window::Theme::Dark => egui::Theme::Dark,
                }),
                events: root_events,
                ..Default::default()
            }
        };

        self.integration.pre_update();
        let mut full_output = self
            .integration
            .update(self.app.as_mut(), None, raw_input, true);

        if let Some(error) = self.shared.borrow_mut().fatal_error.take() {
            return Err(error);
        }

        let mut repaint_root = false;
        {
            let mut shared = self.shared.borrow_mut();
            if let Some(root) = shared.viewports.get_mut(&ViewportId::ROOT) {
                root.info.events.clear();
            }

            shared
                .root_platform
                .handle_platform_output(&full_output.platform_output);
            shared.painter.update_textures(&full_output.textures_delta);

            let commands = full_output
                .viewport_output
                .entries
                .get_mut(&ViewportId::ROOT)
                .map(|root_output| std::mem::take(&mut root_output.commands))
                .unwrap_or_default();
            repaint_root |= shared.process_logical_root_commands(commands);

            let SharedState {
                egui_ctx,
                viewports,
                painter,
                viewport_from_window,
                ..
            } = &mut *shared;
            handle_viewport_output(
                egui_ctx,
                &full_output.viewport_output,
                viewports,
                painter,
                viewport_from_window,
            );

            for viewport in viewports.values_mut() {
                if viewport.ids.this == ViewportId::ROOT {
                    continue;
                }
                let was_uninitialized = viewport.window.is_none();
                viewport.initialize_window(event_loop, egui_ctx, viewport_from_window, painter)?;
                if was_uninitialized && let Some(window) = &viewport.window {
                    window.request_redraw();
                }
            }
        }

        self.integration.maybe_autosave(self.app.as_mut(), None);

        if self.integration.should_close() {
            Ok(EventResult::CloseRequestedAndExit)
        } else if repaint_root {
            Ok(EventResult::RepaintLogicalRootNow)
        } else {
            Ok(EventResult::Wait)
        }
    }

    /// This is called both for the root viewport, and all deferred viewports
    fn run_ui_and_paint(
        &mut self,
        window_id: WindowId,
        event_loop: &ActiveEventLoop,
    ) -> Result<EventResult> {
        profiling::function_scope!();

        let Some(viewport_id) = self
            .shared
            .borrow()
            .viewport_from_window
            .get(&window_id)
            .copied()
        else {
            return Ok(EventResult::Wait);
        };

        profiling::finish_frame!();

        let logical_root = self.logical_root;
        let Self {
            app,
            integration,
            shared,
            ..
        } = self;

        let mut frame_timer = crate::stopwatch::Stopwatch::new();
        frame_timer.start();

        let (viewport_ui_cb, raw_input, is_visible, run_ui) = {
            profiling::scope!("Prepare");
            let mut shared_lock = shared.borrow_mut();

            #[cfg(target_os = "macos")]
            shared_lock.finish_ended_native_resize();

            let SharedState {
                viewports, painter, ..
            } = &mut *shared_lock;

            if viewport_id != ViewportId::ROOT {
                let Some(viewport) = viewports.get(&viewport_id) else {
                    return Ok(EventResult::Wait);
                };

                if viewport.viewport_ui_cb.is_none() {
                    // This will only happen if this is an immediate viewport.
                    // That means that the viewport cannot be rendered by itself and needs his parent to be rendered.
                    if let Some(viewport) = viewports.get(&viewport.ids.parent)
                        && let Some(window) = viewport.window.as_ref()
                    {
                        return Ok(EventResult::RepaintNext(window.id()));
                    }
                    return Ok(EventResult::Wait);
                }
            }

            let Some(viewport) = viewports.get_mut(&viewport_id) else {
                return Ok(EventResult::Wait);
            };

            let Viewport {
                viewport_ui_cb,
                window,
                egui_winit,
                info,
                has_presented,
                ..
            } = viewport;

            let viewport_ui_cb = viewport_ui_cb.clone();

            let Some(window) = window else {
                return Ok(EventResult::Wait);
            };
            egui_winit::update_viewport_info(info, &integration.egui_ctx, window, false);

            // New children are deliberately native-hidden until this pass has painted and
            // presented them. Their actual visibility therefore cannot gate their first frame.
            let is_visible = info.visible().unwrap_or(true) || !*has_presented;

            {
                profiling::scope!("set_window");
                pollster::block_on(painter.set_window(viewport_id, Some(Arc::clone(window))))?;
            }

            let Some(egui_winit) = egui_winit.as_mut() else {
                return Ok(EventResult::Wait);
            };
            let mut raw_input = egui_winit.take_egui_input(window);

            let run_ui = is_visible || is_viewport_or_descendant_visible(viewports, viewport_id);

            integration.pre_update();

            raw_input.time = Some(integration.beginning.elapsed().as_secs_f64());
            raw_input.viewports = viewports
                .iter()
                .map(|(id, viewport)| (*id, viewport.info.clone()))
                .collect();

            painter.handle_screenshots(&mut raw_input.events);

            (viewport_ui_cb, raw_input, is_visible, run_ui)
        };

        // ------------------------------------------------------------

        // Runs the update, which could call immediate viewports,
        // so make sure we hold no locks here!
        let full_output =
            integration.update(app.as_mut(), viewport_ui_cb.as_deref(), raw_input, run_ui);

        if let Some(error) = shared.borrow_mut().fatal_error.take() {
            return Err(error);
        }

        // ------------------------------------------------------------

        let mut shared_mut = shared.borrow_mut();

        let SharedState {
            egui_ctx,
            viewports,
            painter,
            viewport_from_window,
            ..
        } = &mut *shared_mut;

        let FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = full_output;

        let Some(viewport) = viewports.get_mut(&viewport_id) else {
            return Ok(EventResult::Wait);
        };

        viewport.info.events.clear(); // they should have been processed

        let Viewport {
            window: Some(window),
            egui_winit: Some(egui_winit),
            requested_visible,
            requested_active,
            pending_focus,
            has_presented,
            ..
        } = viewport
        else {
            return Ok(EventResult::Wait);
        };

        egui_winit.handle_platform_output_with_event_loop(window, event_loop, platform_output);

        let vsync_secs = if is_visible {
            let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);

            let mut screenshot_commands = vec![];
            viewport.actions_requested.retain(|cmd| {
                if let ActionRequested::Screenshot(info) = cmd {
                    screenshot_commands.push(info.clone());
                    false
                } else {
                    true
                }
            });
            let vsync_secs = painter.paint_and_update_textures(
                viewport_id,
                pixels_per_point,
                app.clear_color(&egui_ctx.global_style().visuals),
                &clipped_primitives,
                &textures_delta,
                screenshot_commands,
                window,
            );

            for action in viewport.actions_requested.drain(..) {
                match action {
                    ActionRequested::Screenshot { .. } => {
                        // already handled above
                    }
                    ActionRequested::Cut => {
                        egui_winit.egui_input_mut().events.push(egui::Event::Cut);
                    }
                    ActionRequested::Copy => {
                        egui_winit.egui_input_mut().events.push(egui::Event::Copy);
                    }
                    ActionRequested::Paste => {
                        if let Some(contents) = egui_winit.clipboard_text() {
                            let contents = contents.replace("\r\n", "\n");
                            if !contents.is_empty() {
                                egui_winit
                                    .egui_input_mut()
                                    .events
                                    .push(egui::Event::Paste(contents));
                            }
                        }
                    }
                }
            }

            if !logical_root {
                integration.post_rendering(window);
            }

            if !*has_presented {
                *has_presented = true;
                window.set_visible(*requested_visible);
                if *requested_visible && (*requested_active || *pending_focus) {
                    window.focus_window();
                    *pending_focus = false;
                } else if *requested_active {
                    *pending_focus = true;
                }
            }

            vsync_secs
        } else {
            painter.update_textures(&textures_delta);
            0.0
        };

        handle_viewport_output(
            &integration.egui_ctx,
            &viewport_output,
            viewports,
            painter,
            viewport_from_window,
        );

        let window = viewport_from_window
            .get(&window_id)
            .and_then(|id| viewports.get(id))
            .and_then(|vp| vp.window.as_ref());

        integration.report_frame_time(frame_timer.total_time_sec() - vsync_secs); // don't count auto-save time as part of regular frame time

        integration.maybe_autosave(app.as_mut(), window.map(|w| w.as_ref()));

        if let Some(window) = window
            && is_invisible_or_minimized(window)
        {
            // On Mac, a minimized Window uses up all CPU:
            // https://github.com/emilk/egui/issues/325
            // On Windows, an invisible window also uses up all CPU:
            // https://github.com/emilk/egui/issues/7776
            profiling::scope!("minimized_sleep");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        if integration.should_close() {
            Ok(EventResult::CloseRequested)
        } else {
            Ok(EventResult::Wait)
        }
    }

    fn on_window_event(
        &mut self,
        window_id: WindowId,
        event: &winit::event::WindowEvent,
    ) -> EventResult {
        let Self {
            integration,
            shared,
            ..
        } = self;
        let mut shared = shared.borrow_mut();

        let viewport_id = shared.viewport_from_window.get(&window_id).copied();

        // On Windows, if a window is resized by the user, it should repaint synchronously, inside the
        // event handler. If this is not done, the compositor will assume that the window does not want
        // to redraw and continue ahead.
        //
        // In eframe's case, that causes the window to rapidly flicker, as it struggles to deliver
        // new frames to the compositor in time. The flickering is technically glutin or glow's fault, but we should be responding properly
        // to resizes anyway, as doing so avoids dropping frames.
        //
        // See: https://github.com/emilk/egui/issues/903
        let mut repaint_asap = false;

        // On macOS the drawn frames must be synchronized with the CoreAnimation transactions
        // driving actual AppKit live resize and fullscreen transitions. Initial and programmatic
        // `Resized` events are not native resize transitions and must not enable transaction
        // presentation.
        #[cfg(target_os = "macos")]
        shared.finish_ended_native_resize();

        #[cfg(not(target_os = "macos"))]
        if !matches!(event, winit::event::WindowEvent::Resized(_))
            && shared
                .native_resize
                .is_some_and(|resize| Some(resize.viewport_id) == viewport_id)
        {
            shared.set_native_resize(None);
        }

        match event {
            winit::event::WindowEvent::Focused(focused) => {
                let focused = if cfg!(target_os = "macos")
                    && let Some(viewport_id) = viewport_id
                    && let Some(viewport) = shared.viewports.get(&viewport_id)
                    && let Some(window) = &viewport.window
                {
                    // TODO(emilk): remove this work-around once we update winit
                    // https://github.com/rust-windowing/winit/issues/4371
                    // https://github.com/emilk/egui/issues/7588
                    window.has_focus()
                } else {
                    *focused
                };

                shared.focused_viewport = focused.then_some(viewport_id).flatten();
                if focused {
                    for viewport in shared.viewports.values_mut() {
                        viewport.currently_focused = false;
                    }
                    if let Some(viewport_id) = viewport_id {
                        let ordinal = shared.next_focus_ordinal;
                        shared.next_focus_ordinal += 1;
                        if let Some(viewport) = shared.viewports.get_mut(&viewport_id) {
                            viewport.currently_focused = true;
                            viewport.focus_ordinal = Some(ordinal);
                        }
                    }
                } else if let Some(viewport_id) = viewport_id
                    && let Some(viewport) = shared.viewports.get_mut(&viewport_id)
                {
                    viewport.currently_focused = false;
                }
            }

            winit::event::WindowEvent::Resized(physical_size) => {
                // Resize with 0 width and height is used by winit to signal a minimize event on Windows.
                // See: https://github.com/rust-windowing/winit/issues/208
                // This solves an issue where the app would panic when minimizing on Windows.
                if let Some(id) = viewport_id
                    && let (Some(width), Some(height)) = (
                        NonZeroU32::new(physical_size.width),
                        NonZeroU32::new(physical_size.height),
                    )
                {
                    #[cfg(target_os = "macos")]
                    let native_resize_state = shared
                        .viewports
                        .get(&id)
                        .and_then(|viewport| viewport.window.as_deref())
                        .map_or(
                            egui_wgpu::winit::NativeResizeState::Idle,
                            native_resize_state,
                        );

                    #[cfg(not(target_os = "macos"))]
                    let native_resize_state = egui_wgpu::winit::NativeResizeState::LiveResize;

                    if native_resize_state != egui_wgpu::winit::NativeResizeState::Idle {
                        shared.set_native_resize(Some(NativeResize {
                            viewport_id: id,
                            state: native_resize_state,
                        }));
                    } else if shared
                        .native_resize
                        .is_some_and(|resize| resize.viewport_id == id)
                    {
                        shared.set_native_resize(None);
                    }

                    shared.painter.on_window_resized(id, width, height);
                    repaint_asap = true;
                }
            }

            winit::event::WindowEvent::Occluded(is_occluded) => {
                if let Some(viewport_id) = viewport_id
                    && let Some(viewport) = shared.viewports.get_mut(&viewport_id)
                {
                    viewport.info.occluded = Some(*is_occluded);
                }
            }

            winit::event::WindowEvent::CloseRequested => {
                if viewport_id == Some(ViewportId::ROOT) && integration.should_close() {
                    log::debug!(
                        "Received WindowEvent::CloseRequested for main viewport - shutting down."
                    );
                    return EventResult::CloseRequested;
                }

                log::debug!("Received WindowEvent::CloseRequested for viewport {viewport_id:?}");

                if let Some(viewport_id) = viewport_id
                    && let Some(viewport) = shared.viewports.get_mut(&viewport_id)
                {
                    // Tell viewport it should close:
                    viewport.info.events.push(egui::ViewportEvent::Close);

                    // We may need to repaint both us and our parent to close the window,
                    // and perhaps twice (once to notice the close-event, once again to enforce it).
                    // `request_repaint_of` does a double-repaint though:
                    integration.egui_ctx.request_repaint_of(viewport_id);
                    integration.egui_ctx.request_repaint_of(viewport.ids.parent);
                }
            }

            _ => {}
        }

        let event_response = viewport_id
            .and_then(|viewport_id| {
                let viewport = shared.viewports.get_mut(&viewport_id)?;
                Some(integration.on_window_event(
                    viewport.window.as_deref()?,
                    viewport.egui_winit.as_mut()?,
                    event,
                ))
            })
            .unwrap_or_default();

        if integration.should_close() {
            EventResult::CloseRequested
        } else if event_response.repaint {
            if repaint_asap {
                EventResult::RepaintNow(window_id)
            } else {
                EventResult::RepaintNext(window_id)
            }
        } else {
            EventResult::Wait
        }
    }
}

impl Viewport {
    /// Create winit window, if needed.
    fn initialize_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        egui_ctx: &egui::Context,
        windows_id: &mut HashMap<WindowId, ViewportId>,
        painter: &mut egui_wgpu::winit::Painter,
    ) -> crate::Result<()> {
        if self.window.is_some() {
            return Ok(()); // we already have one
        }

        profiling::function_scope!();

        let viewport_id = self.ids.this;

        let native_builder = self.builder.clone().with_visible(false).with_active(false);
        let window = Arc::new(egui_winit::create_window(
            egui_ctx,
            event_loop,
            &native_builder,
        )?);
        pollster::block_on(painter.set_window_with_transparency(
            viewport_id,
            Some(Arc::clone(&window)),
            self.builder.transparent.unwrap_or(false),
        ))?;

        self.egui_winit = Some(egui_winit::State::new(
            egui_ctx.clone(),
            viewport_id,
            event_loop,
            Some(window.scale_factor() as f32),
            event_loop.system_theme(),
            painter.max_texture_side(),
        ));

        egui_winit::update_viewport_info(&mut self.info, egui_ctx, &window, true);
        egui_winit::process_viewport_commands(
            egui_ctx,
            &mut self.info,
            std::mem::take(&mut self.deferred_commands),
            &window,
            &mut self.actions_requested,
        );
        windows_id.insert(window.id(), viewport_id);
        self.window = Some(window);
        Ok(())
    }
}

fn create_window(
    egui_ctx: &egui::Context,
    event_loop: &ActiveEventLoop,
    storage: Option<&dyn Storage>,
    native_options: &mut NativeOptions,
) -> Result<(Window, ViewportBuilder), winit::error::OsError> {
    profiling::function_scope!();

    let window_settings = epi_integration::load_window_settings(storage);
    let viewport_builder = epi_integration::viewport_builder(
        egui_ctx.zoom_factor(),
        event_loop,
        native_options,
        window_settings,
    )
    .with_visible(false); // Start hidden until we render the first frame to fix white flash on startup (https://github.com/emilk/egui/pull/3631)

    let window = egui_winit::create_window(egui_ctx, event_loop, &viewport_builder)?;
    epi_integration::apply_window_settings(&window, window_settings);
    Ok((window, viewport_builder))
}

/// Is this viewport, or any of its (transitive) descendant viewports, visible?
///
/// Immediate viewports are rendered inline while their parent's UI runs, so even
/// if this viewport's window is occluded or minimized we must still run its UI to
/// give any visible descendant a chance to be painted.
fn is_viewport_or_descendant_visible(viewports: &Viewports, viewport_id: ViewportId) -> bool {
    let Some(viewport) = viewports.get(&viewport_id) else {
        return false;
    };
    if viewport.info.visible().unwrap_or(true) {
        return true;
    }
    viewports.values().any(|child| {
        child.ids.parent == viewport_id
            && child.ids.this != viewport_id
            && is_viewport_or_descendant_visible(viewports, child.ids.this)
    })
}

/// Chooses the child that paces the logical root and returns its clamped refresh interval.
fn logical_root_repaint_interval(viewports: &Viewports) -> std::time::Duration {
    let eligible = viewports
        .values()
        .filter(|viewport| {
            viewport.ids.this != ViewportId::ROOT
                && viewport.requested_visible
                && !viewport.info.minimized.unwrap_or(false)
                && !viewport.info.occluded.unwrap_or(false)
        })
        .max_by_key(|viewport| {
            (
                viewport.currently_focused,
                viewport.focus_ordinal.unwrap_or(0),
                Reverse(viewport.declaration_ordinal),
            )
        });
    let Some(viewport) = eligible else {
        return std::time::Duration::from_millis(100);
    };

    let refresh_hz = viewport
        .window
        .as_ref()
        .and_then(|window| window.current_monitor())
        .and_then(|monitor| monitor.refresh_rate_millihertz())
        .map_or(60.0, |millihertz| millihertz as f64 / 1_000.0)
        .clamp(30.0, 240.0);
    std::time::Duration::from_secs_f64(1.0 / refresh_hz)
}

fn render_immediate_viewport(
    beginning: Instant,
    shared: &RefCell<SharedState>,
    immediate_viewport: ImmediateViewport<'_>,
) {
    profiling::function_scope!();

    let ImmediateViewport {
        ids,
        declaration_ordinal,
        builder,
        mut viewport_ui_cb,
    } = immediate_viewport;

    let mut initialization_error = None;
    let creation_failed = {
        let mut shared = shared.borrow_mut();
        if shared.fatal_error.is_some() {
            true
        } else {
            let SharedState {
                egui_ctx,
                viewports,
                painter,
                viewport_from_window,
                ..
            } = &mut *shared;
            let viewport = initialize_or_update_viewport(
                viewports,
                ids,
                declaration_ordinal,
                ViewportClass::Immediate,
                builder.clone(),
                None,
                painter,
            );
            if viewport.window.is_none() {
                let result = event_loop_context::with_current_event_loop(|event_loop| {
                    viewport.initialize_window(event_loop, egui_ctx, viewport_from_window, painter)
                });
                match result {
                    Some(Ok(())) => {}
                    Some(Err(error)) => initialization_error = Some(error),
                    None => {
                        initialization_error = Some(crate::Error::UnsupportedConfiguration(
                            "immediate viewport creation requires an active event loop".to_owned(),
                        ));
                    }
                }
            }
            viewport.window.is_none() || initialization_error.is_some()
        }
    };

    if let Some(error) = initialization_error {
        let mut shared = shared.borrow_mut();
        if shared.fatal_error.is_none() {
            shared.fatal_error = Some(error);
        }
    }

    if creation_failed {
        let shared = shared.borrow();
        let egui_ctx = shared.egui_ctx.clone();
        let input = egui::RawInput {
            viewport_id: ids.this,
            viewports: shared
                .viewports
                .iter()
                .map(|(id, viewport)| (*id, viewport.info.clone()))
                .collect(),
            max_texture_side: shared.painter.max_texture_side(),
            focused: false,
            ..Default::default()
        };
        drop(shared);
        let _ = egui_ctx.run_ui(input, |ui| viewport_ui_cb(ui));
        return;
    }

    let input = {
        let SharedState {
            egui_ctx,
            viewports,
            painter,
            ..
        } = &mut *shared.borrow_mut();

        let viewport = initialize_or_update_viewport(
            viewports,
            ids,
            declaration_ordinal,
            ViewportClass::Immediate,
            builder,
            None,
            painter,
        );
        let (Some(window), Some(egui_winit)) = (&viewport.window, &mut viewport.egui_winit) else {
            return;
        };
        egui_winit::update_viewport_info(&mut viewport.info, egui_ctx, window, false);

        let mut input = egui_winit.take_egui_input(window);
        input.viewports = viewports
            .iter()
            .map(|(id, viewport)| (*id, viewport.info.clone()))
            .collect();
        input.time = Some(beginning.elapsed().as_secs_f64());
        input
    };

    let egui_ctx = shared.borrow().egui_ctx.clone();

    // ------------------------------------------

    // Run the user code, which could re-entrantly call this function again (!).
    // Make sure no locks are held during this call.
    let egui::FullOutput {
        platform_output,
        textures_delta,
        shapes,
        pixels_per_point,
        viewport_output,
    } = egui_ctx.run_ui(input, |ui| {
        viewport_ui_cb(ui);
    });

    // ------------------------------------------

    let mut shared_mut = shared.borrow_mut();
    let SharedState {
        viewports,
        painter,
        viewport_from_window,
        ..
    } = &mut *shared_mut;

    let Some(viewport) = viewports.get_mut(&ids.this) else {
        return;
    };
    viewport.info.events.clear(); // they should have been processed
    let (Some(egui_winit), Some(window)) = (&mut viewport.egui_winit, &viewport.window) else {
        return;
    };

    {
        profiling::scope!("set_window");
        if let Err(err) = pollster::block_on(painter.set_window(ids.this, Some(Arc::clone(window))))
        {
            log::error!(
                "when rendering viewport_id={:?}, set_window Error {err}",
                ids.this
            );
        }
    }

    let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);
    painter.paint_and_update_textures(
        ids.this,
        pixels_per_point,
        [0.0, 0.0, 0.0, 0.0],
        &clipped_primitives,
        &textures_delta,
        vec![],
        window,
    );

    if !viewport.has_presented {
        viewport.has_presented = true;
        window.set_visible(viewport.requested_visible);
        if viewport.requested_visible && viewport.requested_active {
            window.focus_window();
        }
    }

    egui_winit.handle_platform_output(window, platform_output);

    handle_viewport_output(
        &egui_ctx,
        &viewport_output,
        viewports,
        painter,
        viewport_from_window,
    );
}

fn remove_viewports_not_in(
    viewports: &mut Viewports,
    painter: &mut egui_wgpu::winit::Painter,
    viewport_from_window: &mut HashMap<WindowId, ViewportId>,
    viewport_output: &OrderedViewportIdMap<ViewportOutput>,
) {
    let active_viewports_ids: ViewportIdSet = viewport_output.keys().copied().collect();

    // Prune dead viewports:
    // Drop renderer-owned surfaces before dropping the native windows they reference.
    painter.gc_viewports(&active_viewports_ids);
    viewports.retain(|id, _| active_viewports_ids.contains(id));
    viewport_from_window.retain(|_, id| active_viewports_ids.contains(id));
}

/// Add new viewports, and update existing ones:
fn handle_viewport_output(
    egui_ctx: &egui::Context,
    viewport_output: &egui::ViewportOutputReport,
    viewports: &mut Viewports,
    painter: &mut egui_wgpu::winit::Painter,
    viewport_from_window: &mut HashMap<WindowId, ViewportId>,
) {
    for (
        viewport_id,
        ViewportOutput {
            declaration_ordinal,
            parent,
            class,
            builder,
            viewport_ui_cb,
            mut commands,
            repaint_delay: _, // ignored - we listened to the repaint callback instead
            requested_repaint_delay: _,
        },
    ) in viewport_output.entries.clone()
    {
        if declaration_ordinal.is_none() {
            continue;
        }
        let ids = ViewportIdPair::from_self_and_parent(viewport_id, parent);

        let viewport = initialize_or_update_viewport(
            viewports,
            ids,
            declaration_ordinal.expect("declared viewport must have an ordinal"),
            class,
            builder,
            viewport_ui_cb,
            painter,
        );

        for command in &commands {
            winit_integration::apply_stateful_viewport_command_to_builder(
                &mut viewport.builder,
                command,
            );
            match command {
                egui::ViewportCommand::Visible(visible) => {
                    viewport.requested_visible = *visible;
                }
                egui::ViewportCommand::Transparent(transparent) => {
                    if let Err(err) = painter.set_surface_transparency(viewport_id, *transparent) {
                        log::error!("Failed to update viewport transparency: {err}");
                    }
                }
                egui::ViewportCommand::Icon(icon) => {
                    viewport.icon_state = ViewportIconState::Explicit(icon.clone());
                }
                _ => {}
            }
        }
        viewport.deferred_commands.append(&mut commands);
        if viewport.window.is_none() {
            let state = winit_integration::fold_pre_creation_viewport_commands(
                &mut viewport.builder,
                &mut viewport.deferred_commands,
            );
            viewport.pending_focus |= state.pending_focus;
            viewport.requested_visible = viewport.builder.visible.unwrap_or(true);
            viewport.requested_active = viewport.builder.active.unwrap_or(true);
        }
        if let Some(window) = viewport.window.as_ref() {
            let old_inner_size = window.inner_size();

            egui_winit::process_viewport_commands(
                egui_ctx,
                &mut viewport.info,
                std::mem::take(&mut viewport.deferred_commands),
                window,
                &mut viewport.actions_requested,
            );

            if viewport.has_presented && viewport.requested_visible && viewport.pending_focus {
                window.focus_window();
                viewport.pending_focus = false;
            }

            // For Wayland : https://github.com/emilk/egui/issues/4196
            if cfg!(target_os = "linux") {
                let new_inner_size = window.inner_size();
                if new_inner_size != old_inner_size
                    && let (Some(width), Some(height)) = (
                        NonZeroU32::new(new_inner_size.width),
                        NonZeroU32::new(new_inner_size.height),
                    )
                {
                    painter.on_window_resized(viewport_id, width, height);
                }
            }
        }
    }

    if viewport_output.is_complete {
        remove_viewports_not_in(
            viewports,
            painter,
            viewport_from_window,
            &viewport_output.entries,
        );
    }
}

fn initialize_or_update_viewport<'a>(
    viewports: &'a mut Viewports,
    ids: ViewportIdPair,
    declaration_ordinal: u64,
    class: ViewportClass,
    mut builder: ViewportBuilder,
    viewport_ui_cb: Option<Arc<dyn Fn(&mut egui::Ui) + Send + Sync>>,
    painter: &mut egui_wgpu::winit::Painter,
) -> &'a mut Viewport {
    use std::collections::btree_map::Entry;

    profiling::function_scope!();

    let declared_icon = builder.icon.clone();
    let inherited_icon = viewports
        .get(&ids.parent)
        .and_then(|viewport| viewport.builder.icon.clone());

    match viewports.entry(ids.this) {
        Entry::Vacant(entry) => {
            // New viewport:
            let icon_state = declared_icon.map_or(ViewportIconState::Inherited, |icon| {
                ViewportIconState::Explicit(Some(icon))
            });
            if matches!(icon_state, ViewportIconState::Inherited) {
                builder.icon = inherited_icon;
            }
            log::debug!("Creating new viewport {:?} ({:?})", ids.this, builder.title);
            let requested_visible = builder.visible.unwrap_or(true);
            let requested_active = builder.active.unwrap_or(true);
            entry.insert(Viewport {
                ids,
                declaration_ordinal,
                class,
                builder,
                icon_state,
                deferred_commands: vec![],
                info: Default::default(),
                actions_requested: Vec::new(),
                viewport_ui_cb,
                window: None,
                egui_winit: None,
                requested_visible,
                requested_active,
                pending_focus: false,
                has_presented: false,
                currently_focused: false,
                focus_ordinal: None,
            })
        }

        Entry::Occupied(mut entry) => {
            // Patch an existing viewport:
            let viewport = entry.get_mut();

            viewport.class = class;
            viewport.ids.parent = ids.parent;
            debug_assert_eq!(viewport.declaration_ordinal, declaration_ordinal);
            // Child viewport passes can report an existing deferred viewport without its
            // root-owned callback. Keep the callback so direct repaints still run that viewport.
            if let Some(viewport_ui_cb) = viewport_ui_cb {
                viewport.viewport_ui_cb = Some(viewport_ui_cb);
            }

            if let Some(visible) = builder.visible {
                viewport.requested_visible = visible;
            }
            if let Some(active) = builder.active {
                viewport.requested_active = active;
            }
            if let Some(icon) = declared_icon {
                viewport.icon_state = ViewportIconState::Explicit(Some(icon.clone()));
                builder.icon = Some(icon);
            } else {
                builder.icon = match &viewport.icon_state {
                    ViewportIconState::Inherited => inherited_icon,
                    ViewportIconState::Explicit(icon) => icon.clone(),
                };
            }
            let (mut delta_commands, recreate) = viewport.builder.patch(builder);

            if recreate {
                log::debug!(
                    "Recreating window for viewport {:?} ({:?})",
                    ids.this,
                    viewport.builder.title
                );
                viewport.window = None;
                viewport.egui_winit = None;
                viewport.has_presented = false;
                viewport.currently_focused = false;
                painter.remove_surface(viewport.ids.this);
            }

            viewport.deferred_commands.append(&mut delta_commands);

            entry.into_mut()
        }
    }
}
