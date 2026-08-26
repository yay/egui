//! Note that this file contains code very similar to [`super::wgpu_integration`].
//! When making changes to one you often also want to apply it to the other.
//!
//! This is also very complex code, and not very pretty.
//! There is a bunch of improvements we could do,
//! like removing a bunch of `unwraps`.

#![expect(clippy::undocumented_unsafe_blocks)]
#![expect(clippy::unwrap_used)]

use std::{
    cell::RefCell,
    cmp::Reverse,
    num::NonZeroU32,
    rc::Rc,
    sync::{Arc, atomic::AtomicBool, atomic::Ordering},
    time::Instant,
};

use egui_winit::ActionRequested;
use glow::HasContext as _;
use glutin::{
    context::NotCurrentGlContext as _,
    display::GetGlDisplay as _,
    prelude::{GlDisplay as _, PossiblyCurrentGlContext as _},
    surface::GlSurface as _,
};
use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
use winit::{
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

use ahash::HashMap;
use egui::{
    DeferredViewportUiCallback, ImmediateViewport, OrderedViewportIdMap, ViewportBuilder,
    ViewportClass, ViewportId, ViewportIdPair, ViewportInfo, ViewportOutput,
};
#[cfg(feature = "accesskit")]
use egui_winit::accesskit_winit;

use crate::{
    App, AppCreator, CreationContext, NativeOptions, Result, Storage,
    native::{epi_integration::EpiIntegration, winit_integration::is_invisible_or_minimized},
};

use super::{
    epi_integration, event_loop_context,
    winit_integration::{EventResult, UserEvent, WinitApp, create_egui_context},
};

// ----------------------------------------------------------------------------
// Types:

/// Set only when both glutin detachment and platform-native recovery fail.
static GLOW_REUSE_POISONED: AtomicBool = AtomicBool::new(false);

pub struct GlowWinitApp<'app> {
    repaint_proxy: Arc<egui::mutex::Mutex<EventLoopProxy<UserEvent>>>,
    app_name: String,
    native_options: NativeOptions,
    running: Option<GlowWinitRunning<'app>>,

    // Note that since this `AppCreator` is FnOnce we are currently unable to support
    // re-initializing the `GlowWinitRunning` state on Android if the application
    // suspends and resumes.
    app_creator: Option<AppCreator<'app>>,

    /// An optional pre-existing egui context. If `Some`, it is used instead of
    /// creating a new one via [`create_egui_context`]. Taken during initialization.
    egui_ctx: Option<egui::Context>,
}

/// State that is initialized when the application is first starts running via
/// a Resumed event. On Android this ensures that any graphics state is only
/// initialized once the application has an associated `SurfaceView`.
struct GlowWinitRunning<'app> {
    integration: EpiIntegration,
    app: Box<dyn 'app + App>,
    logical_root: bool,

    // These needs to be shared with the immediate viewport renderer, hence the Rc/Arc/RefCells:
    glutin: Rc<RefCell<GlutinWindowContext>>,

    // NOTE: one painter shared by all viewports.
    painter: Rc<RefCell<egui_glow::Painter>>,
}

/// This struct will contain both persistent and temporary glutin state.
///
/// Platform Quirks:
/// * Microsoft Windows: requires that we create a window before opengl context.
/// * Android: window and surface should be destroyed when we receive a suspend event. recreate on resume event.
///
/// winit guarantees that we will get a Resumed event on startup on all platforms.
/// * Before Resumed event: `gl_config`, `gl_context` can be created at any time. on windows, a window must be created to get `gl_context`.
/// * Resumed: `gl_surface` will be created here. `window` will be re-created here for android.
/// * Suspended: on android, we drop window + surface.  on other platforms, we don't get Suspended event.
///
/// The setup is divided between the `new` fn and `on_resume` fn. we can just assume that `on_resume` is a continuation of
/// `new` fn on all platforms. only on android, do we get multiple resumed events because app can be suspended.
struct GlutinWindowContext {
    egui_ctx: egui::Context,

    swap_interval: glutin::surface::SwapInterval,
    gl_config: glutin::config::Config,

    max_texture_side: Option<usize>,

    current_gl_context: Option<glutin::context::PossiblyCurrentContext>,
    not_current_gl_context: Option<glutin::context::NotCurrentContext>,

    /// Persistent controller drawable on Windows. macOS uses a surfaceless CGL context.
    controller_surface: Option<glutin::surface::Surface<glutin::surface::PbufferSurface>>,
    logical_root: bool,
    transparency_supported: bool,
    root_platform: egui_winit::WindowIndependentState,
    root_events: Vec<egui::Event>,
    #[cfg(feature = "accesskit")]
    active_accesskit_windows: ahash::HashSet<WindowId>,
    next_focus_ordinal: u64,

    /// First backend failure raised inside an immediate viewport callback.
    ///
    /// Immediate viewport APIs are infallible, so the enclosing dispatch returns this after the
    /// callback has received its required synthetic result.
    fatal_error: Option<crate::Error>,

    viewports: OrderedViewportIdMap<Viewport>,
    viewport_from_window: HashMap<WindowId, ViewportId>,
    window_from_viewport: OrderedViewportIdMap<WindowId>,

    focused_viewport: Option<ViewportId>,
}

#[derive(Clone)]
enum ViewportIconState {
    Inherited,
    Explicit(Option<Arc<egui::IconData>>),
}

struct Viewport {
    ids: ViewportIdPair,
    declaration_ordinal: u64,
    class: ViewportClass,
    builder: ViewportBuilder,
    icon_state: ViewportIconState,
    deferred_commands: Vec<egui::viewport::ViewportCommand>,
    info: ViewportInfo,
    actions_requested: Vec<egui_winit::ActionRequested>,

    /// The user-callback that shows the ui.
    /// None for immediate viewports.
    viewport_ui_cb: Option<Arc<DeferredViewportUiCallback>>,

    // These three live and die together.
    // TODO(emilk): clump them together into one struct!
    gl_surface: Option<glutin::surface::Surface<glutin::surface::WindowSurface>>,
    window: Option<Arc<Window>>,
    egui_winit: Option<egui_winit::State>,
    requested_visible: bool,
    requested_active: bool,
    /// Focus action retained until a requested-visible child has presented its first frame.
    pending_focus: bool,
    has_presented: bool,
    currently_focused: bool,
    focus_ordinal: Option<u64>,
}

// ----------------------------------------------------------------------------

impl<'app> GlowWinitApp<'app> {
    pub fn new(
        event_loop: &EventLoop<UserEvent>,
        app_name: &str,
        native_options: NativeOptions,
        egui_ctx: Option<egui::Context>,
        app_creator: AppCreator<'app>,
    ) -> Self {
        profiling::function_scope!();
        Self {
            repaint_proxy: Arc::new(egui::mutex::Mutex::new(event_loop.create_proxy())),
            app_name: app_name.to_owned(),
            native_options,
            running: None,
            app_creator: Some(app_creator),
            egui_ctx,
        }
    }

    #[expect(unsafe_code)]
    fn create_glutin_windowed_context(
        egui_ctx: &egui::Context,
        event_loop: &ActiveEventLoop,
        storage: Option<&dyn Storage>,
        native_options: &mut NativeOptions,
    ) -> Result<(GlutinWindowContext, egui_glow::Painter)> {
        profiling::function_scope!();
        let logical_root = native_options.root_viewport_mode == crate::RootViewportMode::Windowless;
        let window_settings = (!logical_root)
            .then(|| epi_integration::load_window_settings(storage))
            .flatten();
        let winit_window_builder = if logical_root {
            if native_options.window_builder.take().is_some() {
                log::warn!("`NativeOptions::window_builder` is ignored for a windowless root");
            }
            native_options
                .viewport
                .clone()
                .with_visible(false)
                .with_inner_size([1.0, 1.0])
                .with_active(false)
                .with_decorations(false)
                .with_taskbar(false)
        } else {
            epi_integration::viewport_builder(
                egui_ctx.zoom_factor(),
                event_loop,
                native_options,
                window_settings,
            )
            .with_visible(false)
        };

        let mut glutin_window_context = unsafe {
            GlutinWindowContext::new(
                egui_ctx,
                winit_window_builder,
                native_options,
                event_loop,
                logical_root,
            )?
        };

        if !logical_root {
            // Creates the root surface and winit input state.
            if let Err(error) =
                glutin_window_context.initialize_window(ViewportId::ROOT, event_loop)
            {
                let _detached = glutin_window_context.prepare_for_drop();
                return Err(error);
            }
        }

        if !logical_root {
            let viewport = &glutin_window_context.viewports[&ViewportId::ROOT];
            let window = viewport.window.as_ref().unwrap(); // Can't fail - we just called `initialize_all_viewports`
            epi_integration::apply_window_settings(window, window_settings);
        }

        let gl = unsafe {
            profiling::scope!("glow::Context::from_loader_function");
            Arc::new(glow::Context::from_loader_function(|s| {
                let s = std::ffi::CString::new(s)
                    .expect("failed to construct C string from string for gl proc address");

                glutin_window_context.get_proc_address(&s)
            }))
        };

        let painter = match egui_glow::Painter::new(
            gl,
            "",
            native_options.glow_options.shader_version,
            native_options.dithering,
        ) {
            Ok(painter) => painter,
            Err(error) => {
                let _detached = glutin_window_context.prepare_for_drop();
                return Err(error.into());
            }
        };

        Ok((glutin_window_context, painter))
    }

    fn init_run_state(
        &mut self,
        event_loop: &ActiveEventLoop,
    ) -> Result<&mut GlowWinitRunning<'app>> {
        profiling::function_scope!();

        if GLOW_REUSE_POISONED.load(Ordering::Acquire) {
            return Err(crate::Error::UnsupportedConfiguration(
                "native GL state could not be detached after an earlier Glow run".to_owned(),
            ));
        }

        if self.native_options.root_viewport_mode == crate::RootViewportMode::Windowless
            && !cfg!(any(target_os = "macos", target_os = "windows"))
        {
            return Err(crate::Error::UnsupportedConfiguration(
                "windowless root requires macOS or Windows".to_owned(),
            ));
        }

        if self.native_options.root_viewport_mode == crate::RootViewportMode::Windowless {
            epi_integration::ensure_default_egui_icon(&mut self.native_options.viewport);
        }

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
            .unwrap_or_else(|| create_egui_context(storage.as_deref()));

        let (mut glutin, painter) = Self::create_glutin_windowed_context(
            &egui_ctx,
            event_loop,
            storage.as_deref(),
            &mut self.native_options,
        )?;
        let gl = Arc::clone(painter.gl());

        let max_texture_side = painter.max_texture_side();
        glutin.max_texture_side = Some(max_texture_side);
        for viewport in glutin.viewports.values_mut() {
            if let Some(egui_winit) = viewport.egui_winit.as_mut() {
                egui_winit.set_max_texture_side(max_texture_side);
            }
        }

        let painter = Rc::new(RefCell::new(painter));

        let logical_root =
            self.native_options.root_viewport_mode == crate::RootViewportMode::Windowless;
        let root_window = glutin.window_opt(ViewportId::ROOT);
        let integration = EpiIntegration::new(
            egui_ctx,
            root_window.as_ref(),
            event_loop.display_handle().map(|handle| handle.as_raw()),
            &self.app_name,
            &self.native_options,
            storage,
            Some(Arc::clone(&gl)),
            Some(Box::new({
                let painter = Rc::clone(&painter);
                move |native| painter.borrow_mut().register_native_texture(native)
            })),
            #[cfg(feature = "wgpu_no_default_features")]
            None,
        );

        {
            let event_loop_proxy = Arc::clone(&self.repaint_proxy);
            integration
                .egui_ctx
                .set_request_repaint_callback(move |info| {
                    log::trace!("request_repaint_callback: {info:?}");
                    let now = Instant::now();
                    let when = now + info.delay;
                    let requested_when = now + info.requested_delay;
                    let cumulative_pass_nr = info.current_cumulative_pass_nr;
                    event_loop_proxy
                        .lock()
                        .send_event(UserEvent::RequestRepaint {
                            viewport_id: info.viewport_id,
                            when,
                            requested_when,
                            genuinely_immediate: info.requested_delay.is_zero(),
                            cumulative_pass_nr,
                        })
                        .ok();
                });
        }

        #[cfg(feature = "accesskit")]
        {
            let event_loop_proxy = self.repaint_proxy.lock().clone();
            let viewport = glutin.viewports.get_mut(&ViewportId::ROOT).unwrap(); // we always have a root
            if let Viewport {
                window: Some(window),
                egui_winit: Some(egui_winit),
                ..
            } = viewport
            {
                egui_winit.init_accesskit(event_loop, window, event_loop_proxy);
            }
        }

        if self
            .native_options
            .viewport
            .mouse_passthrough
            .unwrap_or(false)
            && let Some(root_window) = root_window.as_ref()
            && let Err(err) = root_window.set_cursor_hittest(false)
        {
            log::warn!("set_cursor_hittest(false) failed: {err}");
        }

        let app_creator = std::mem::take(&mut self.app_creator)
            .expect("Single-use AppCreator has unexpectedly already been taken");

        crate::maybe_attach_inspection_plugin(&integration.egui_ctx, Some(self.app_name.clone()));

        let app: Box<dyn 'app + App> = {
            // Use latest raw_window_handle for eframe compatibility
            use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};

            let gl_config = glutin.gl_config.clone();
            let get_proc_address = move |addr: &_| gl_config.display().get_proc_address(addr);
            let window = glutin.window_opt(ViewportId::ROOT);
            let cc = CreationContext {
                egui_ctx: integration.egui_ctx.clone(),
                integration_info: integration.frame.info().clone(),
                storage: integration.frame.storage(),
                gl: Some(gl),
                get_proc_address: Some(Arc::new(get_proc_address)),
                #[cfg(feature = "wgpu_no_default_features")]
                wgpu_render_state: None,
                window: window.clone(),
                raw_display_handle: event_loop.display_handle().map(|h| h.as_raw()),
                raw_window_handle: window.as_ref().map_or(
                    Err(raw_window_handle::HandleError::NotSupported),
                    |window| window.window_handle().map(|h| h.as_raw()),
                ),
            };
            profiling::scope!("app_creator");
            match app_creator(&cc) {
                Ok(app) => app,
                Err(err) => {
                    painter.borrow_mut().destroy();
                    let _detached = glutin.prepare_for_drop();
                    return Err(crate::Error::AppCreation(err));
                }
            }
        };

        let glutin = Rc::new(RefCell::new(glutin));

        {
            // Create weak pointers so that we don't keep
            // state alive for too long.
            let glutin = Rc::downgrade(&glutin);
            let painter = Rc::downgrade(&painter);
            let beginning = integration.beginning;

            egui::Context::set_immediate_viewport_renderer(move |egui_ctx, immediate_viewport| {
                if let (Some(glutin), Some(painter)) = (glutin.upgrade(), painter.upgrade()) {
                    render_immediate_viewport(
                        egui_ctx,
                        &glutin,
                        &painter,
                        beginning,
                        immediate_viewport,
                    );
                } else {
                    log::warn!("render_sync_callback called after window closed");
                }
            });
        }

        Ok(self.running.insert(GlowWinitRunning {
            integration,
            app,
            glutin,
            painter,
            logical_root,
        }))
    }
}

impl WinitApp for GlowWinitApp<'_> {
    fn egui_ctx(&self) -> Option<&egui::Context> {
        self.running.as_ref().map(|r| &r.integration.egui_ctx)
    }

    fn window(&self, window_id: WindowId) -> Option<Arc<Window>> {
        let running = self.running.as_ref()?;
        let glutin = running.glutin.borrow();
        let viewport_id = *glutin.viewport_from_window.get(&window_id)?;
        if let Some(viewport) = glutin.viewports.get(&viewport_id) {
            viewport.window.clone()
        } else {
            None
        }
    }

    fn window_id_from_viewport_id(&self, id: ViewportId) -> Option<WindowId> {
        self.running
            .as_ref()?
            .glutin
            .borrow()
            .window_from_viewport
            .get(&id)
            .copied()
    }

    fn save(&mut self) {
        log::debug!("WinitApp::save called");
        if let Some(running) = self.running.as_mut() {
            profiling::function_scope!();

            // This is used because of the "save on suspend" logic on Android. Once the application is suspended, there is no window associated to it, which was causing panics when `.window().expect()` was used.
            let window_opt = running.glutin.borrow().window_opt(ViewportId::ROOT);

            running
                .integration
                .save(running.app.as_mut(), window_opt.as_deref());
        }
    }

    fn save_and_destroy(&mut self) {
        if let Some(mut running) = self.running.take() {
            profiling::function_scope!();

            let root_window = running.glutin.borrow().window_opt(ViewportId::ROOT);
            running
                .integration
                .save(running.app.as_mut(), root_window.as_deref());
            let controller_ready = !running.logical_root
                || running
                    .glutin
                    .borrow_mut()
                    .make_controller_current()
                    .is_ok();
            if controller_ready {
                running.app.on_exit(Some(running.painter.borrow().gl()));
                running.painter.borrow_mut().destroy();
            } else {
                running.app.on_exit(None);
                running.painter.borrow_mut().abandon_without_gl_cleanup();
            }
            let _detached = running.glutin.borrow_mut().prepare_for_drop();
        }
    }

    fn run_ui_and_paint(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
    ) -> Result<EventResult> {
        if let Some(running) = &mut self.running {
            running.run_ui_and_paint(event_loop, window_id)
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
                logical_root_repaint_interval(&running.glutin.borrow().viewports)
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

        let running = if let Some(running) = &mut self.running {
            // Not the first resume event. Create all outstanding windows.
            running
                .glutin
                .borrow_mut()
                .initialize_all_windows(event_loop)?;
            running
        } else {
            // First resume event. Create our root window etc.
            self.init_run_state(event_loop)?
        };
        if running.logical_root {
            Ok(EventResult::RepaintLogicalRootNow)
        } else {
            let window_id = running.glutin.borrow().window_from_viewport[&ViewportId::ROOT];
            Ok(EventResult::RepaintNow(window_id))
        }
    }

    fn suspended(&mut self, _: &ActiveEventLoop) -> crate::Result<EventResult> {
        if let Some(running) = &mut self.running {
            running.glutin.borrow_mut().on_suspend()?;
        }
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
            let mut glutin = running.glutin.borrow_mut();
            if let Some(viewport) = glutin
                .focused_viewport
                .and_then(|viewport| glutin.viewports.get_mut(&viewport))
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
        _: &ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) -> Result<EventResult> {
        if let Some(running) = &mut self.running {
            let result = running.on_window_event(window_id, &event);
            if let Some(error) = running.glutin.borrow_mut().fatal_error.take() {
                Err(error)
            } else {
                Ok(result)
            }
        } else {
            Ok(EventResult::Exit)
        }
    }

    #[cfg(feature = "accesskit")]
    fn on_accesskit_event(&mut self, event: accesskit_winit::Event) -> crate::Result<EventResult> {
        if let Some(running) = &self.running {
            let mut glutin = running.glutin.borrow_mut();
            let viewport_id = glutin.viewport_from_window.get(&event.window_id).copied();
            match &event.window_event {
                accesskit_winit::WindowEvent::InitialTreeRequested => {
                    let was_empty = glutin.active_accesskit_windows.is_empty();
                    glutin.active_accesskit_windows.insert(event.window_id);
                    if was_empty {
                        running.integration.egui_ctx.enable_accesskit();
                    }
                    return Ok(EventResult::RepaintNow(event.window_id));
                }
                accesskit_winit::WindowEvent::AccessibilityDeactivated => {
                    glutin.active_accesskit_windows.remove(&event.window_id);
                    if glutin.active_accesskit_windows.is_empty() {
                        running.integration.egui_ctx.disable_accesskit();
                    }
                    return Ok(EventResult::Wait);
                }
                accesskit_winit::WindowEvent::ActionRequested(request) => {
                    if let Some(viewport) = viewport_id.and_then(|id| glutin.viewports.get_mut(&id))
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

impl GlowWinitRunning<'_> {
    /// Runs one application pass while the logical root has no native window.
    fn run_logical_root(&mut self, event_loop: &ActiveEventLoop) -> Result<EventResult> {
        profiling::function_scope!();

        self.glutin.borrow_mut().make_controller_current()?;
        let raw_input = {
            let mut glutin = self.glutin.borrow_mut();
            let commands = glutin
                .viewports
                .get_mut(&ViewportId::ROOT)
                .map(|root| std::mem::take(&mut root.deferred_commands))
                .unwrap_or_default();
            let _ = glutin.process_logical_root_commands(commands);
            let events = std::mem::take(&mut glutin.root_events);
            let predicted_dt = logical_root_repaint_interval(&glutin.viewports).as_secs_f32();
            egui::RawInput {
                viewport_id: ViewportId::ROOT,
                viewports: glutin
                    .viewports
                    .iter()
                    .map(|(id, viewport)| (*id, viewport.info.clone()))
                    .collect(),
                max_texture_side: glutin.max_texture_side,
                predicted_dt,
                focused: false,
                system_theme: event_loop.system_theme().map(|theme| match theme {
                    winit::window::Theme::Light => egui::Theme::Light,
                    winit::window::Theme::Dark => egui::Theme::Dark,
                }),
                events,
                ..Default::default()
            }
        };

        self.integration.pre_update();
        let mut full_output = self
            .integration
            .update(self.app.as_mut(), None, raw_input, true);

        if let Some(error) = self.glutin.borrow_mut().fatal_error.take() {
            return Err(error);
        }

        let mut repaint_root = false;
        {
            let mut glutin = self.glutin.borrow_mut();
            glutin.make_controller_current()?;
            if let Some(root) = glutin.viewports.get_mut(&ViewportId::ROOT) {
                root.info.events.clear();
            }
            glutin
                .root_platform
                .handle_platform_output(&full_output.platform_output);

            let mut painter = self.painter.borrow_mut();
            for (id, image_delta) in &full_output.textures_delta.set {
                painter.set_texture(*id, image_delta);
            }
            for id in &full_output.textures_delta.free {
                painter.free_texture(*id);
            }
            drop(painter);

            let commands = full_output
                .viewport_output
                .entries
                .get_mut(&ViewportId::ROOT)
                .map(|root_output| std::mem::take(&mut root_output.commands))
                .unwrap_or_default();
            repaint_root |= glutin.process_logical_root_commands(commands);

            glutin.handle_viewport_output(
                event_loop,
                &self.integration.egui_ctx,
                &full_output.viewport_output,
            )?;
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

    #[expect(unsafe_code)]
    fn run_ui_and_paint(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
    ) -> Result<EventResult> {
        profiling::function_scope!();
        let logical_root = self.logical_root;

        let Some(viewport_id) = self
            .glutin
            .borrow()
            .viewport_from_window
            .get(&window_id)
            .copied()
        else {
            return Ok(EventResult::Wait);
        };

        profiling::finish_frame!();

        let mut frame_timer = crate::stopwatch::Stopwatch::new();
        frame_timer.start();

        {
            let glutin = self.glutin.borrow();
            let viewport = &glutin.viewports[&viewport_id];
            let is_immediate = viewport.viewport_ui_cb.is_none();
            if is_immediate && viewport_id != ViewportId::ROOT {
                // This will only happen if this is an immediate viewport.
                // That means that the viewport cannot be rendered by itself and needs his parent to be rendered.
                if let Some(parent_viewport) = glutin.viewports.get(&viewport.ids.parent)
                    && let Some(window) = parent_viewport.window.as_ref()
                {
                    return Ok(EventResult::RepaintNext(window.id()));
                }
                return Ok(EventResult::Wait);
            }
        }

        let (raw_input, viewport_ui_cb, is_visible, run_ui) = {
            let mut glutin = self.glutin.borrow_mut();
            let egui_ctx = glutin.egui_ctx.clone();
            let Some(viewport) = glutin.viewports.get_mut(&viewport_id) else {
                return Ok(EventResult::Wait);
            };
            let Some(window) = viewport.window.as_ref() else {
                return Ok(EventResult::Wait);
            };
            egui_winit::update_viewport_info(&mut viewport.info, &egui_ctx, window, false);

            // New children are deliberately native-hidden until this pass has painted and
            // presented them. Their actual visibility therefore cannot gate their first frame.
            let is_visible = viewport.info.visible().unwrap_or(true) || !viewport.has_presented;

            let Some(egui_winit) = viewport.egui_winit.as_mut() else {
                return Ok(EventResult::Wait);
            };
            let mut raw_input = egui_winit.take_egui_input(window);
            let viewport_ui_cb = viewport.viewport_ui_cb.clone();

            let run_ui =
                is_visible || is_viewport_or_descendant_visible(&glutin.viewports, viewport_id);

            self.integration.pre_update();

            raw_input.time = Some(self.integration.beginning.elapsed().as_secs_f64());
            raw_input.viewports = glutin
                .viewports
                .iter()
                .map(|(id, viewport)| (*id, viewport.info.clone()))
                .collect();

            (raw_input, viewport_ui_cb, is_visible, run_ui)
        };

        // HACK: In order to get the right clear_color, the system theme needs to be set, which
        // usually only happens in the `update` call. So we call Options::begin_pass early
        // to set the right theme. Without this there would be a black flash on the first frame.
        self.integration
            .egui_ctx
            .options_mut(|opt| opt.begin_pass(&raw_input));
        let clear_color = self
            .app
            .clear_color(&self.integration.egui_ctx.global_style().visuals);

        let has_many_viewports = self.glutin.borrow().viewports.len() > 1;
        let clear_before_update = !has_many_viewports; // HACK: for some reason, an early clear doesn't "take" on Mac with multiple viewports.

        if is_visible && clear_before_update {
            // clear before we call update, so users can paint between clear-color and egui windows:

            let mut glutin = self.glutin.borrow_mut();
            let GlutinWindowContext {
                viewports,
                current_gl_context,
                not_current_gl_context,
                ..
            } = &mut *glutin;
            let viewport = &viewports[&viewport_id];
            let Some(window) = viewport.window.as_ref() else {
                return Ok(EventResult::Wait);
            };
            let Some(gl_surface) = viewport.gl_surface.as_ref() else {
                return Ok(EventResult::Wait);
            };

            let screen_size_in_pixels: [u32; 2] = window.inner_size().into();

            {
                frame_timer.pause();
                change_gl_context(current_gl_context, not_current_gl_context, gl_surface)?;
                unsafe {
                    self.painter
                        .borrow()
                        .gl()
                        .bind_framebuffer(glow::FRAMEBUFFER, None);
                }
                frame_timer.resume();
            }

            self.painter
                .borrow()
                .clear(screen_size_in_pixels, clear_color);
        }

        // ------------------------------------------------------------
        // The update function, which could call immediate viewports,
        // so make sure we don't hold any locks here required by the immediate viewports rendeer.

        let full_output = self.integration.update(
            self.app.as_mut(),
            viewport_ui_cb.as_deref(),
            raw_input,
            run_ui,
        );

        if let Some(error) = self.glutin.borrow_mut().fatal_error.take() {
            return Err(error);
        }

        // ------------------------------------------------------------

        let Self {
            integration,
            app,
            glutin,
            painter,
            ..
        } = self;

        let mut glutin = glutin.borrow_mut();
        let mut painter = painter.borrow_mut();

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = full_output;

        let GlutinWindowContext {
            viewports,
            current_gl_context,
            not_current_gl_context,
            ..
        } = &mut *glutin;

        let Some(viewport) = viewports.get_mut(&viewport_id) else {
            return Ok(EventResult::Wait);
        };

        viewport.info.events.clear(); // they should have been processed
        let window = viewport.window.clone().unwrap();
        let gl_surface = viewport.gl_surface.as_ref().unwrap();
        let egui_winit = viewport.egui_winit.as_mut().unwrap();

        egui_winit.handle_platform_output_with_event_loop(&window, event_loop, platform_output);

        // Upload textures even when not visible: the atlas dirty region is already
        // consumed, so dropping the delta would desync the font texture.
        let has_texture_updates = !textures_delta.set.is_empty() || !textures_delta.free.is_empty();
        if is_visible || has_texture_updates {
            // We may need to switch contexts again, because of immediate viewports:
            frame_timer.pause();
            change_gl_context(current_gl_context, not_current_gl_context, gl_surface)?;
            unsafe {
                painter.gl().bind_framebuffer(glow::FRAMEBUFFER, None);
            }
            frame_timer.resume();
        }

        for (id, image_delta) in &textures_delta.set {
            painter.set_texture(*id, image_delta);
        }

        if is_visible {
            let clipped_primitives = integration.egui_ctx.tessellate(shapes, pixels_per_point);

            let screen_size_in_pixels: [u32; 2] = window.inner_size().into();

            if !clear_before_update {
                painter.clear(screen_size_in_pixels, clear_color);
            }

            painter.paint_primitives(screen_size_in_pixels, pixels_per_point, &clipped_primitives);

            {
                for action in viewport.actions_requested.drain(..) {
                    match action {
                        ActionRequested::Screenshot(user_data) => {
                            let screenshot = painter.read_screen_rgba(screen_size_in_pixels);
                            egui_winit
                                .egui_input_mut()
                                .events
                                .push(egui::Event::Screenshot {
                                    viewport_id,
                                    user_data,
                                    image: screenshot.into(),
                                });
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
                    integration.post_rendering(&window);
                }
            }

            {
                // vsync - don't count as frame-time:
                frame_timer.pause();
                profiling::scope!("swap_buffers");
                let context = current_gl_context.as_ref().ok_or_else(|| {
                    egui_glow::PainterError::from(
                        "failed to get current context to swap buffers".to_owned(),
                    )
                })?;

                gl_surface.swap_buffers(context)?;
                frame_timer.resume();
            }

            if !viewport.has_presented {
                viewport.has_presented = true;
                window.set_visible(viewport.requested_visible);
                if viewport.requested_visible
                    && (viewport.requested_active || viewport.pending_focus)
                {
                    window.focus_window();
                    viewport.pending_focus = false;
                } else if viewport.requested_active {
                    viewport.pending_focus = true;
                }
            }

            // give it time to settle:
            #[cfg(feature = "__screenshot")]
            if integration.egui_ctx.cumulative_pass_nr() == 2
                && let Ok(path) = std::env::var("EFRAME_SCREENSHOT_TO")
            {
                save_screenshot_and_exit(&path, &painter, screen_size_in_pixels);
            }
        }

        // Free textures *after* painting, since they may still be used in the frame we just drew.
        for id in &textures_delta.free {
            painter.free_texture(*id);
        }

        glutin.handle_viewport_output(event_loop, &integration.egui_ctx, &viewport_output)?;

        integration.report_frame_time(frame_timer.total_time_sec()); // don't count auto-save time as part of regular frame time

        integration.maybe_autosave(app.as_mut(), Some(&window));

        if is_invisible_or_minimized(&window) {
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
        let mut glutin = self.glutin.borrow_mut();
        let viewport_id = glutin.viewport_from_window.get(&window_id).copied();

        // On Windows, if a window is resized by the user, it should repaint synchronously, inside the
        // event handler.
        //
        // If this is not done, the compositor will assume that the window does not want to redraw,
        // and continue ahead.
        //
        // In eframe's case, that causes the window to rapidly flicker, as it struggles to deliver
        // new frames to the compositor in time.
        //
        // The flickering is technically glutin or glow's fault, but we should be responding properly
        // to resizes anyway, as doing so avoids dropping frames.
        //
        // See: https://github.com/emilk/egui/issues/903
        let mut repaint_asap = false;

        match event {
            winit::event::WindowEvent::Focused(focused) => {
                let focused = if cfg!(target_os = "macos")
                    && let Some(viewport_id) = viewport_id
                    && let Some(viewport) = glutin.viewports.get(&viewport_id)
                    && let Some(window) = &viewport.window
                {
                    // TODO(emilk): remove this work-around once we update winit
                    // https://github.com/rust-windowing/winit/issues/4371
                    // https://github.com/emilk/egui/issues/7588
                    window.has_focus()
                } else {
                    *focused
                };

                glutin.focused_viewport = focused.then_some(viewport_id).flatten();
                if focused {
                    for viewport in glutin.viewports.values_mut() {
                        viewport.currently_focused = false;
                    }
                    if let Some(viewport_id) = viewport_id {
                        let ordinal = glutin.next_focus_ordinal;
                        glutin.next_focus_ordinal += 1;
                        if let Some(viewport) = glutin.viewports.get_mut(&viewport_id) {
                            viewport.currently_focused = true;
                            viewport.focus_ordinal = Some(ordinal);
                        }
                    }
                } else if let Some(viewport_id) = viewport_id
                    && let Some(viewport) = glutin.viewports.get_mut(&viewport_id)
                {
                    viewport.currently_focused = false;
                }
            }

            winit::event::WindowEvent::Resized(physical_size) => {
                // Resize with 0 width and height is used by winit to signal a minimize event on Windows.
                // See: https://github.com/rust-windowing/winit/issues/208
                // This solves an issue where the app would panic when minimizing on Windows.
                if 0 < physical_size.width
                    && 0 < physical_size.height
                    && let Some(viewport_id) = viewport_id
                {
                    repaint_asap = true;
                    glutin.resize(viewport_id, *physical_size);
                }
            }

            winit::event::WindowEvent::Occluded(is_occluded) => {
                if let Some(viewport_id) = viewport_id
                    && let Some(viewport) = glutin.viewports.get_mut(&viewport_id)
                {
                    viewport.info.occluded = Some(*is_occluded);
                }
            }

            winit::event::WindowEvent::CloseRequested => {
                if viewport_id == Some(ViewportId::ROOT) && self.integration.should_close() {
                    log::debug!(
                        "Received WindowEvent::CloseRequested for main viewport - shutting down."
                    );
                    return EventResult::CloseRequested;
                }

                log::debug!("Received WindowEvent::CloseRequested for viewport {viewport_id:?}");

                if let Some(viewport_id) = viewport_id
                    && let Some(viewport) = glutin.viewports.get_mut(&viewport_id)
                {
                    // Tell viewport it should close:
                    viewport.info.events.push(egui::ViewportEvent::Close);

                    // We may need to repaint both us and our parent to close the window,
                    // and perhaps twice (once to notice the close-event, once again to enforce it).
                    // `request_repaint_of` does a double-repaint though:
                    self.integration.egui_ctx.request_repaint_of(viewport_id);
                    self.integration
                        .egui_ctx
                        .request_repaint_of(viewport.ids.parent);
                }
            }
            _ => {}
        }

        if self.integration.should_close() {
            return EventResult::CloseRequested;
        }

        let mut event_response = egui_winit::EventResponse {
            consumed: false,
            repaint: false,
        };
        if let Some(viewport_id) = viewport_id {
            if let Some(viewport) = glutin.viewports.get_mut(&viewport_id) {
                if let (Some(window), Some(egui_winit)) =
                    (&viewport.window, &mut viewport.egui_winit)
                {
                    event_response = self.integration.on_window_event(window, egui_winit, event);
                }
            } else {
                log::trace!("Ignoring event: no viewport for {viewport_id:?}");
            }
        } else {
            log::trace!("Ignoring event: no viewport_id");
        }

        if event_response.repaint {
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

fn change_gl_context(
    current_gl_context: &mut Option<glutin::context::PossiblyCurrentContext>,
    not_current_gl_context: &mut Option<glutin::context::NotCurrentContext>,
    gl_surface: &glutin::surface::Surface<glutin::surface::WindowSurface>,
) -> Result {
    profiling::function_scope!();

    if !cfg!(any(target_os = "macos", target_os = "windows")) {
        // According to https://github.com/emilk/egui/issues/4289
        // We cannot do this early-out on Windows. A surfaceless CGL controller also has no
        // current view for `Surface::is_current` to query, so macOS takes the safe path too.
        // TODO(emilk): optimize context switching on Windows and macOS too.
        // See https://github.com/emilk/egui/issues/4173

        if let Some(current_gl_context) = current_gl_context {
            profiling::scope!("is_current");
            if gl_surface.is_current(current_gl_context) {
                return Ok(()); // Early-out to save a lot of time.
            }
        }
    }

    profiling::scope!("make_current");
    if let Some(current) = current_gl_context.as_ref() {
        current.make_current(gl_surface)?;
    } else {
        let not_current = not_current_gl_context
            .take()
            .expect("GL context is neither current nor available");
        *current_gl_context = Some(not_current.make_current(gl_surface)?);
    }
    Ok(())
}

/// Last-resort native detachment used only after glutin failed to clear its current context.
#[expect(unsafe_code)]
fn clear_current_gl_context_natively() -> bool {
    #[cfg(target_os = "macos")]
    unsafe {
        #[link(name = "OpenGL", kind = "framework")]
        unsafe extern "C" {
            fn CGLSetCurrentContext(context: *mut std::ffi::c_void) -> i32;
            fn CGLGetCurrentContext() -> *mut std::ffi::c_void;
        }

        CGLSetCurrentContext(std::ptr::null_mut()) == 0 && CGLGetCurrentContext().is_null()
    }

    #[cfg(target_os = "windows")]
    unsafe {
        #[link(name = "opengl32")]
        unsafe extern "system" {
            fn wglMakeCurrent(
                device_context: *mut std::ffi::c_void,
                rendering_context: *mut std::ffi::c_void,
            ) -> i32;
            fn wglGetCurrentContext() -> *mut std::ffi::c_void;
        }

        wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut()) != 0
            && wglGetCurrentContext().is_null()
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        false
    }
}

impl GlutinWindowContext {
    /// Applies commands addressed to the permanently windowless root.
    ///
    /// Commands can arrive while a child callback is running, so they remain queued until the
    /// controller's next pass. The return value requests that pass when new root input was added.
    fn process_logical_root_commands(&mut self, commands: Vec<egui::ViewportCommand>) -> bool {
        let root = self
            .viewports
            .get_mut(&ViewportId::ROOT)
            .expect("logical root viewport must exist");
        let state = crate::native::winit_integration::process_logical_root_commands(
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

    /// Detaches the context, then drops surfaces/windows before dropping the context itself.
    ///
    /// Returns `false` only if both glutin and platform-native detachment fail. In that case a
    /// subsequent Glow run in this process is rejected because reusable native state is unknown.
    fn prepare_for_drop(&mut self) -> bool {
        let detached = self.current_gl_context.as_ref().is_none_or(|context| {
            if context.make_not_current_in_place().is_ok() {
                true
            } else {
                let recovered = clear_current_gl_context_natively();
                if !recovered {
                    log::error!("Failed to detach the native GL context during cleanup");
                }
                recovered
            }
        });

        for viewport in self.viewports.values_mut() {
            viewport.gl_surface = None;
            viewport.egui_winit = None;
            viewport.window = None;
        }
        self.controller_surface = None;
        self.current_gl_context = None;
        self.not_current_gl_context = None;

        if !detached {
            GLOW_REUSE_POISONED.store(true, Ordering::Release);
        }
        detached
    }

    /// Makes the persistent windowless controller target current.
    #[allow(unreachable_patterns)]
    fn make_controller_current(&mut self) -> Result {
        if !self.logical_root {
            return Ok(());
        }

        #[cfg(target_os = "windows")]
        {
            let surface = self.controller_surface.as_ref().ok_or_else(|| {
                crate::Error::UnsupportedConfiguration(
                    "windowless WGL controller pbuffer is missing".to_owned(),
                )
            })?;
            if let Some(context) = self.current_gl_context.as_ref() {
                context.make_current(surface)?;
            } else if let Some(context) = self.not_current_gl_context.take() {
                self.current_gl_context = Some(context.make_current(surface)?);
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Some(context) = self.current_gl_context.as_ref() {
                match context {
                    glutin::context::PossiblyCurrentContext::Cgl(context) => {
                        context.make_current_surfaceless()?;
                    }
                    _ => {
                        return Err(crate::Error::UnsupportedConfiguration(
                            "windowless Glow on macOS requires CGL".to_owned(),
                        ));
                    }
                }
            } else if let Some(context) = self.not_current_gl_context.take() {
                self.current_gl_context = Some(match context {
                    glutin::context::NotCurrentContext::Cgl(context) => {
                        glutin::context::PossiblyCurrentContext::Cgl(
                            context.make_current_surfaceless()?,
                        )
                    }
                    _ => {
                        return Err(crate::Error::UnsupportedConfiguration(
                            "windowless Glow on macOS requires CGL".to_owned(),
                        ));
                    }
                });
            }
        }

        Ok(())
    }

    #[expect(unsafe_code)]
    #[allow(unreachable_patterns)]
    unsafe fn new(
        egui_ctx: &egui::Context,
        viewport_builder: ViewportBuilder,
        native_options: &NativeOptions,
        event_loop: &ActiveEventLoop,
        logical_root: bool,
    ) -> Result<Self> {
        profiling::function_scope!();

        // There is a lot of complexity with opengl creation,
        // so prefer extensive logging to get all the help we can to debug issues.

        use glutin::prelude::*;
        // convert native options to glutin options
        let hardware_acceleration = match native_options.glow_options.hardware_acceleration {
            egui_glow::HardwareAcceleration::Required => Some(true),
            egui_glow::HardwareAcceleration::Preferred => None,
            egui_glow::HardwareAcceleration::Off => Some(false),
        };
        let swap_interval = if native_options.glow_options.vsync {
            glutin::surface::SwapInterval::Wait(NonZeroU32::MIN)
        } else {
            glutin::surface::SwapInterval::DontWait
        };
        /*  opengl setup flow goes like this:
            1. we create a configuration for opengl "Display" / "Config" creation
            2. choose between special extensions like glx or egl or wgl and use them to create config/display
            3. opengl context configuration
            4. opengl context creation
        */
        // start building config for gl display
        let config_template_builder = glutin::config::ConfigTemplateBuilder::new()
            .prefer_hardware_accelerated(hardware_acceleration)
            .with_depth_size(native_options.depth_buffer)
            .with_stencil_size(native_options.stencil_buffer)
            // A windowless controller has no authoritative root surface. Prefer an alpha-capable
            // config so later opaque and transparent children can coexist on the shared context.
            .with_transparency(
                logical_root || native_options.viewport.transparent.unwrap_or(false),
            );
        // we don't know if multi sampling option is set. so, check if its more than 0.
        let config_template_builder = if native_options.multisampling > 0 {
            config_template_builder.with_multisampling(
                native_options
                    .multisampling
                    .try_into()
                    .expect("failed to fit multisamples option of native_options into u8"),
            )
        } else {
            config_template_builder
        };

        log::debug!("trying to create glutin Display with config: {config_template_builder:?}");

        // Create GL display. This may probably create a window too on most platforms. Definitely on `MS windows`. Never on Android.
        let helper_window_attributes = if logical_root && cfg!(target_os = "macos") {
            None
        } else {
            Some(egui_winit::create_winit_window_attributes(
                egui_ctx,
                viewport_builder.clone(),
            ))
        };
        let build_display = |template: glutin::config::ConfigTemplateBuilder| {
            glutin_winit::DisplayBuilder::new()
                // The FallbackEgl rationale is documented in egui pull request 2526.
                .with_preference(glutin_winit::ApiPreference::FallbackEgl)
                .with_window_attributes(helper_window_attributes.clone())
                .build(event_loop, template, |mut config_iterator| {
                    let config = config_iterator.next().expect(
                        "failed to find a matching configuration for creating glutin config",
                    );
                    log::debug!(
                        "using the first config from config picker closure. config: {config:?}"
                    );
                    config
                })
        };

        let (window, gl_config) = {
            profiling::scope!("DisplayBuilder::build");

            match build_display(config_template_builder.clone()) {
                Ok(result) => result,
                Err(first_error) if logical_root => {
                    log::warn!(
                        "No alpha-capable GL config was available for the windowless root: \
                         {first_error}. Retrying with opaque-only child support."
                    );
                    let opaque_template = config_template_builder.clone().with_transparency(false);
                    build_display(opaque_template.clone()).map_err(|error| {
                        crate::Error::NoGlutinConfigs(opaque_template.build(), error)
                    })?
                }
                Err(error) => {
                    return Err(crate::Error::NoGlutinConfigs(
                        config_template_builder.build(),
                        error,
                    ));
                }
            }
        };
        if let Some(window) = &window {
            egui_winit::apply_viewport_builder_to_window(egui_ctx, window, &viewport_builder);
        }

        let gl_display = gl_config.display();
        log::debug!(
            "successfully created GL Display with version: {} and supported features: {:?}",
            gl_display.version_string(),
            gl_display.supported_features()
        );
        let glutin_raw_window_handle = window.as_ref().map(|w| {
            w.window_handle()
                .expect("Failed to get window handle")
                .as_raw()
        });
        log::debug!("creating gl context using raw window handle: {glutin_raw_window_handle:?}");

        // create gl context. if core context cannot be created, try gl es context as fallback.
        let context_attributes =
            glutin::context::ContextAttributesBuilder::new().build(glutin_raw_window_handle);
        let fallback_context_attributes = glutin::context::ContextAttributesBuilder::new()
            .with_context_api(glutin::context::ContextApi::Gles(None))
            .build(glutin_raw_window_handle);

        let gl_context_result = unsafe {
            profiling::scope!("create_context");
            gl_config
                .display()
                .create_context(&gl_config, &context_attributes)
        };

        let gl_context = match gl_context_result {
            Ok(it) => it,
            Err(err) => {
                log::warn!(
                    "Failed to create context using default context attributes {context_attributes:?} due to error: {err}"
                );
                log::debug!(
                    "Retrying with fallback context attributes: {fallback_context_attributes:?}"
                );
                unsafe {
                    gl_config
                        .display()
                        .create_context(&gl_config, &fallback_context_attributes)?
                }
            }
        };
        // Cache this while the WGL bootstrap HDC is valid. glutin 0.32.3 retains that HDC in its
        // config, so the windowless path must not query config attributes after dropping the
        // helper HWND.
        let transparency_supported = gl_config.supports_transparency().unwrap_or(false);

        #[cfg(target_os = "windows")]
        let (current_gl_context, not_current_gl_context, controller_surface) = if logical_root {
            let attributes =
                glutin::surface::SurfaceAttributesBuilder::<glutin::surface::PbufferSurface>::new()
                    .build(NonZeroU32::MIN, NonZeroU32::MIN);
            let surface = unsafe {
                gl_config
                    .display()
                    .create_pbuffer_surface(&gl_config, &attributes)?
            };
            let context = gl_context.make_current(&surface)?;
            (Some(context), None, Some(surface))
        } else {
            (None, Some(gl_context), None)
        };

        #[cfg(target_os = "macos")]
        let (current_gl_context, not_current_gl_context, controller_surface) = if logical_root {
            let context = match gl_context {
                glutin::context::NotCurrentContext::Cgl(context) => {
                    glutin::context::PossiblyCurrentContext::Cgl(
                        context.make_current_surfaceless()?,
                    )
                }
                _ => {
                    return Err(crate::Error::UnsupportedConfiguration(
                        "windowless Glow on macOS requires CGL".to_owned(),
                    ));
                }
            };
            (Some(context), None, None)
        } else {
            (None, Some(gl_context), None)
        };

        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let (current_gl_context, not_current_gl_context, controller_surface) =
            (None, Some(gl_context), None);

        let mut viewport_from_window = HashMap::default();
        let mut window_from_viewport = OrderedViewportIdMap::default();
        let mut viewport_info = ViewportInfo::default();
        if !logical_root && let Some(window) = &window {
            viewport_from_window.insert(window.id(), ViewportId::ROOT);
            window_from_viewport.insert(ViewportId::ROOT, window.id());
            egui_winit::update_viewport_info(&mut viewport_info, egui_ctx, window, true);

            // Tell egui right away about native_pixels_per_point etc,
            // so that the app knows about it during app creation:
            let pixels_per_point = egui_winit::pixels_per_point(egui_ctx, window);

            egui_ctx.input_mut(|i| {
                i.raw
                    .viewports
                    .insert(ViewportId::ROOT, viewport_info.clone());

                i.pixels_per_point = pixels_per_point;
            });
        }

        if logical_root {
            viewport_info.title = native_options
                .viewport
                .title
                .clone()
                .or_else(|| viewport_builder.title.clone());
            viewport_info.focused = Some(false);
        }

        let mut viewports = OrderedViewportIdMap::default();
        let root_builder = if logical_root {
            native_options.viewport.clone()
        } else {
            viewport_builder
        };
        viewports.insert(
            ViewportId::ROOT,
            Viewport {
                ids: ViewportIdPair::ROOT,
                declaration_ordinal: 0,
                class: ViewportClass::Root,
                builder: root_builder,
                icon_state: ViewportIconState::Explicit(native_options.viewport.icon.clone()),
                deferred_commands: vec![],
                info: viewport_info,
                actions_requested: Default::default(),
                viewport_ui_cb: None,
                gl_surface: None,
                window: if logical_root {
                    None
                } else {
                    window.map(Arc::new)
                },
                egui_winit: None,
                requested_visible: !logical_root,
                requested_active: !logical_root,
                pending_focus: false,
                has_presented: false,
                currently_focused: !logical_root,
                focus_ordinal: (!logical_root).then_some(0),
            },
        );

        // the fun part with opengl gl is that we never know whether there is an error. the context creation might have failed, but
        // it could keep working until we try to make surface current or swap buffers or something else. future glutin improvements might
        // help us start from scratch again if we fail context creation and go back to preferEgl or try with different config etc..
        // https://github.com/emilk/egui/pull/2541#issuecomment-1370767582

        let mut slf = Self {
            egui_ctx: egui_ctx.clone(),
            swap_interval,
            gl_config,
            current_gl_context,
            not_current_gl_context,
            controller_surface,
            logical_root,
            transparency_supported,
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
            viewports,
            viewport_from_window,
            max_texture_side: None,
            window_from_viewport,
            focused_viewport: (!logical_root).then_some(ViewportId::ROOT),
        };

        if !logical_root {
            slf.initialize_window(ViewportId::ROOT, event_loop)?;
        }

        Ok(slf)
    }

    /// Create a surface, window, and winit integration for all viewports lacking any of that.
    ///
    fn initialize_all_windows(&mut self, event_loop: &ActiveEventLoop) -> Result {
        profiling::function_scope!();

        let viewports: Vec<ViewportId> = self.viewports.keys().copied().collect();

        for viewport_id in viewports {
            if self.logical_root && viewport_id == ViewportId::ROOT {
                continue;
            }
            let was_uninitialized = self
                .viewports
                .get(&viewport_id)
                .is_some_and(|viewport| viewport.window.is_none());
            self.initialize_window(viewport_id, event_loop)?;
            if was_uninitialized
                && let Some(window) = self
                    .viewports
                    .get(&viewport_id)
                    .and_then(|viewport| viewport.window.as_ref())
            {
                window.request_redraw();
            }
        }
        Ok(())
    }

    /// Create a surface, window, and winit integration for the viewport, if missing.
    #[expect(unsafe_code)]
    pub(crate) fn initialize_window(
        &mut self,
        viewport_id: ViewportId,
        event_loop: &ActiveEventLoop,
    ) -> Result {
        profiling::function_scope!();

        let viewport = self
            .viewports
            .get_mut(&viewport_id)
            .expect("viewport doesn't exist");

        let window = if let Some(window) = &mut viewport.window {
            window
        } else {
            log::debug!("Creating a window for viewport {viewport_id:?}");
            let native_builder = viewport
                .builder
                .clone()
                .with_visible(false)
                .with_active(false);
            let window_attributes =
                egui_winit::create_winit_window_attributes(&self.egui_ctx, native_builder.clone());
            if window_attributes.transparent() && !self.transparency_supported {
                log::error!("Cannot create transparent window: the GL config does not support it");
            }
            let window =
                glutin_winit::finalize_window(event_loop, window_attributes, &self.gl_config)?;
            egui_winit::apply_viewport_builder_to_window(&self.egui_ctx, &window, &native_builder);

            egui_winit::update_viewport_info(&mut viewport.info, &self.egui_ctx, &window, true);
            viewport.window.insert(Arc::new(window))
        };

        viewport.egui_winit.get_or_insert_with(|| {
            log::debug!("Initializing egui_winit for viewport {viewport_id:?}");
            egui_winit::State::new(
                self.egui_ctx.clone(),
                viewport_id,
                event_loop,
                Some(window.scale_factor() as f32),
                event_loop.system_theme(),
                self.max_texture_side,
            )
        });

        if viewport.gl_surface.is_none() {
            log::debug!("Creating a gl_surface for viewport {viewport_id:?}");

            // surface attributes
            let (width_px, height_px): (u32, u32) = window.inner_size().into();
            let width_px = NonZeroU32::new(width_px).unwrap_or(NonZeroU32::MIN);
            let height_px = NonZeroU32::new(height_px).unwrap_or(NonZeroU32::MIN);
            let surface_attributes = {
                glutin::surface::SurfaceAttributesBuilder::<glutin::surface::WindowSurface>::new()
                    .build(
                        window
                            .window_handle()
                            .expect("Failed to get display handle")
                            .as_raw(),
                        width_px,
                        height_px,
                    )
            };

            log::trace!("creating surface with attributes: {surface_attributes:?}");
            let gl_surface = unsafe {
                self.gl_config
                    .display()
                    .create_window_surface(&self.gl_config, &surface_attributes)?
            };

            log::trace!("surface created successfully: {gl_surface:?}. making context current");

            if let Some(current_gl_context) = self.current_gl_context.as_ref() {
                current_gl_context.make_current(&gl_surface)?;
            } else {
                let not_current_gl_context = self
                    .not_current_gl_context
                    .take()
                    .expect("GL context is neither current nor available");
                self.current_gl_context = Some(not_current_gl_context.make_current(&gl_surface)?);
            }

            // try setting swap interval. but its not absolutely necessary, so don't panic on failure.
            log::trace!("made context current. setting swap interval for surface");
            if let Err(err) = gl_surface.set_swap_interval(
                self.current_gl_context.as_ref().unwrap(),
                self.swap_interval,
            ) {
                log::warn!("Failed to set swap interval due to error: {err}");
            }

            // we will reach this point only once in most platforms except android.
            // create window/surface/make context current once and just use them forever.

            viewport.gl_surface = Some(gl_surface);
        }

        egui_winit::process_viewport_commands(
            &self.egui_ctx,
            &mut viewport.info,
            std::mem::take(&mut viewport.deferred_commands),
            window,
            &mut viewport.actions_requested,
        );

        self.viewport_from_window.insert(window.id(), viewport_id);
        self.window_from_viewport.insert(viewport_id, window.id());

        Ok(())
    }

    /// only applies for android. but we basically drop surface + window and make context not current
    fn on_suspend(&mut self) -> Result {
        log::debug!("received suspend event. dropping window and surface");
        for viewport in self.viewports.values_mut() {
            viewport.gl_surface = None;
            viewport.window = None;
        }
        if let Some(current) = self.current_gl_context.take() {
            log::debug!("context is current, so making it non-current");
            self.not_current_gl_context = Some(current.make_not_current()?);
        } else {
            log::debug!("context is already not current??? could be duplicate suspend event");
        }
        Ok(())
    }

    fn viewport(&self, viewport_id: ViewportId) -> &Viewport {
        self.viewports
            .get(&viewport_id)
            .expect("viewport doesn't exist")
    }

    fn window_opt(&self, viewport_id: ViewportId) -> Option<Arc<Window>> {
        self.viewport(viewport_id).window.clone()
    }

    fn resize(&mut self, viewport_id: ViewportId, physical_size: winit::dpi::PhysicalSize<u32>) {
        let width_px = NonZeroU32::new(physical_size.width).unwrap_or(NonZeroU32::MIN);
        let height_px = NonZeroU32::new(physical_size.height).unwrap_or(NonZeroU32::MIN);

        if let Some(viewport) = self.viewports.get(&viewport_id)
            && let Some(gl_surface) = &viewport.gl_surface
        {
            if let Err(error) = change_gl_context(
                &mut self.current_gl_context,
                &mut self.not_current_gl_context,
                gl_surface,
            ) {
                if self.fatal_error.is_none() {
                    self.fatal_error = Some(error);
                }
                return;
            }
            gl_surface.resize(
                self.current_gl_context
                    .as_ref()
                    .expect("failed to get current context to resize surface"),
                width_px,
                height_px,
            );
        }
    }

    fn get_proc_address(&self, addr: &std::ffi::CStr) -> *const std::ffi::c_void {
        self.gl_config.display().get_proc_address(addr)
    }

    /// Moves the GL context away from any viewport that is about to be removed.
    ///
    /// Dropping a native window/surface while the GL context is still current to that surface can
    /// crash on Windows. Before pruning stale child viewports, make the context current on a
    /// surviving viewport, preferring the root viewport because it should stay alive for the app
    /// lifetime. If no surviving surface exists, make the context not current.
    fn make_removed_viewports_not_current(
        &mut self,
        viewport_output: &OrderedViewportIdMap<ViewportOutput>,
    ) {
        let removed_current_viewport = self.current_gl_context.as_ref().is_some_and(|current| {
            self.viewports.iter().any(|(id, viewport)| {
                !viewport_output.contains_key(id)
                    && viewport
                        .gl_surface
                        .as_ref()
                        .is_some_and(|surface| surface.is_current(current))
            })
        });
        if !removed_current_viewport {
            return;
        }
        if self.logical_root {
            if let Err(err) = self.make_controller_current() {
                log::warn!("Failed to restore the windowless GL controller target: {err}");
            }
            return;
        }

        let Self {
            viewports,
            current_gl_context,
            not_current_gl_context,
            ..
        } = self;

        if current_gl_context.is_none() {
            return;
        }

        let replacement_surface = viewports
            .get(&ViewportId::ROOT)
            .filter(|_| viewport_output.contains_key(&ViewportId::ROOT))
            .and_then(|viewport| viewport.gl_surface.as_ref())
            .or_else(|| {
                viewports.iter().find_map(|(id, viewport)| {
                    if viewport_output.contains_key(id) {
                        viewport.gl_surface.as_ref()
                    } else {
                        None
                    }
                })
            });

        if let Some(replacement_surface) = replacement_surface {
            if let Err(error) = change_gl_context(
                current_gl_context,
                not_current_gl_context,
                replacement_surface,
            ) {
                if self.fatal_error.is_none() {
                    self.fatal_error = Some(error);
                }
            }
        } else if let Some(current) = current_gl_context.take() {
            match current.make_not_current() {
                Ok(not_current) => {
                    *not_current_gl_context = Some(not_current);
                }
                Err(err) => {
                    log::warn!(
                        "Failed to make GL context not current before removing viewport: {err}"
                    );
                }
            }
        }
    }

    fn remove_viewports_not_in(&mut self, viewport_output: &OrderedViewportIdMap<ViewportOutput>) {
        self.make_removed_viewports_not_current(viewport_output);

        // GC old viewports
        self.viewports
            .retain(|id, _| viewport_output.contains_key(id));
        self.viewport_from_window
            .retain(|_, id| viewport_output.contains_key(id));
        self.window_from_viewport
            .retain(|id, _| viewport_output.contains_key(id));
    }

    fn handle_viewport_output(
        &mut self,
        event_loop: &ActiveEventLoop,
        egui_ctx: &egui::Context,
        viewport_output: &egui::ViewportOutputReport,
    ) -> Result {
        profiling::function_scope!();

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
                &mut self.viewports,
                ids,
                declaration_ordinal.expect("declared viewport must have an ordinal"),
                class,
                builder,
                viewport_ui_cb,
            );

            for command in &commands {
                crate::native::winit_integration::apply_stateful_viewport_command_to_builder(
                    &mut viewport.builder,
                    command,
                );
                match command {
                    egui::ViewportCommand::Visible(visible) => {
                        viewport.requested_visible = *visible;
                    }
                    egui::ViewportCommand::Icon(icon) => {
                        viewport.icon_state = ViewportIconState::Explicit(icon.clone());
                    }
                    _ => {}
                }
            }
            viewport.deferred_commands.append(&mut commands);
            if viewport.window.is_none() {
                let state = crate::native::winit_integration::fold_pre_creation_viewport_commands(
                    &mut viewport.builder,
                    &mut viewport.deferred_commands,
                );
                viewport.pending_focus |= state.pending_focus;
                viewport.requested_visible = viewport.builder.visible.unwrap_or(true);
                viewport.requested_active = viewport.builder.active.unwrap_or(true);
            }
            if let Some(window) = &viewport.window {
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
                    if new_inner_size != old_inner_size {
                        self.resize(viewport_id, new_inner_size);
                    }
                }
            }
        }

        // Create windows for any new viewports:
        self.initialize_all_windows(event_loop)?;

        if viewport_output.is_complete {
            self.remove_viewports_not_in(&viewport_output.entries);
        }
        Ok(())
    }
}

fn initialize_or_update_viewport(
    viewports: &mut OrderedViewportIdMap<Viewport>,
    ids: ViewportIdPair,
    declaration_ordinal: u64,
    class: ViewportClass,
    mut builder: ViewportBuilder,
    viewport_ui_cb: Option<Arc<dyn Fn(&mut egui::Ui) + Send + Sync>>,
) -> &mut Viewport {
    profiling::function_scope!();

    use std::collections::btree_map::Entry;

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
                actions_requested: Default::default(),
                viewport_ui_cb,
                window: None,
                egui_winit: None,
                gl_surface: None,
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

            viewport.ids.parent = ids.parent;
            debug_assert_eq!(viewport.declaration_ordinal, declaration_ordinal);
            viewport.class = class;
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
                viewport.gl_surface = None;
                viewport.has_presented = false;
                viewport.currently_focused = false;
            }

            viewport.deferred_commands.append(&mut delta_commands);

            entry.into_mut()
        }
    }
}

/// Is this viewport, or any of its (transitive) descendant viewports, visible?
///
/// Immediate viewports are rendered inline while their parent's UI runs, so even
/// if this viewport's window is occluded or minimized we must still run its UI to
/// give any visible descendant a chance to be painted.
fn is_viewport_or_descendant_visible(
    viewports: &OrderedViewportIdMap<Viewport>,
    viewport_id: ViewportId,
) -> bool {
    let Some(viewport) = viewports.get(&viewport_id) else {
        return false;
    };
    if viewport.info.visible().unwrap_or(true) {
        return true;
    }
    viewports.values().any(|child| {
        child.ids.parent == viewport_id
            && child.ids.this != viewport_id // ROOT is its own parent; avoid self-recursion.
            && is_viewport_or_descendant_visible(viewports, child.ids.this)
    })
}

/// Chooses the child that paces the logical root and returns its clamped refresh interval.
fn logical_root_repaint_interval(
    viewports: &OrderedViewportIdMap<Viewport>,
) -> std::time::Duration {
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

/// Framebuffer bindings that must survive a nested immediate-viewport dispatch.
#[derive(Clone, Copy)]
enum FramebufferBindings {
    Unified(i32),
    Separate { read: i32, draw: i32 },
}

impl FramebufferBindings {
    /// Captures unified or separate read/draw bindings according to the active GL version.
    #[expect(unsafe_code)]
    fn capture(gl: &glow::Context) -> Self {
        let version = gl.version();
        let separate = version.major >= 3;
        unsafe {
            if separate {
                Self::Separate {
                    read: gl.get_parameter_i32(glow::READ_FRAMEBUFFER_BINDING),
                    draw: gl.get_parameter_i32(glow::DRAW_FRAMEBUFFER_BINDING),
                }
            } else {
                Self::Unified(gl.get_parameter_i32(glow::FRAMEBUFFER_BINDING))
            }
        }
    }

    /// Restores the exact bindings captured before switching native GL targets.
    #[expect(unsafe_code)]
    fn restore(self, gl: &glow::Context) {
        let framebuffer = |value: i32| NonZeroU32::new(value as u32).map(glow::NativeFramebuffer);
        unsafe {
            match self {
                Self::Unified(value) => {
                    gl.bind_framebuffer(glow::FRAMEBUFFER, framebuffer(value));
                }
                Self::Separate { read, draw } => {
                    gl.bind_framebuffer(glow::READ_FRAMEBUFFER, framebuffer(read));
                    gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, framebuffer(draw));
                }
            }
        }
    }
}

/// Restores the parent/controller target and framebuffer state after an immediate child.
fn restore_immediate_parent(
    glutin: &mut GlutinWindowContext,
    painter: &egui_glow::Painter,
    parent: ViewportId,
    framebuffers: FramebufferBindings,
) -> Result {
    if parent == ViewportId::ROOT && glutin.logical_root {
        glutin.make_controller_current()?;
    } else {
        let GlutinWindowContext {
            viewports,
            current_gl_context,
            not_current_gl_context,
            ..
        } = glutin;
        if let Some(parent_surface) = viewports
            .get(&parent)
            .and_then(|viewport| viewport.gl_surface.as_ref())
        {
            change_gl_context(current_gl_context, not_current_gl_context, parent_surface)?;
        }
    }
    framebuffers.restore(painter.gl());
    Ok(())
}

/// This is called (via a callback) by user code to render immediate viewports,
/// i.e. viewport that are directly nested inside a parent viewport.
#[expect(unsafe_code)]
fn render_immediate_viewport(
    egui_ctx: &egui::Context,
    glutin: &RefCell<GlutinWindowContext>,
    painter: &RefCell<egui_glow::Painter>,
    beginning: Instant,
    immediate_viewport: ImmediateViewport<'_>,
) {
    profiling::function_scope!();

    let ImmediateViewport {
        ids,
        declaration_ordinal,
        builder,
        mut viewport_ui_cb,
    } = immediate_viewport;

    let viewport_id = ids.this;
    let previous_framebuffers = FramebufferBindings::capture(painter.borrow().gl());

    let creation_failed = {
        let mut glutin = glutin.borrow_mut();

        if glutin.fatal_error.is_some() {
            true
        } else {
            initialize_or_update_viewport(
                &mut glutin.viewports,
                ids,
                declaration_ordinal,
                ViewportClass::Immediate,
                builder,
                None,
            );

            let ret = event_loop_context::with_current_event_loop(|event_loop| {
                glutin.initialize_window(viewport_id, event_loop)
            });

            match ret {
                Some(Ok(())) => false,
                Some(Err(error)) => {
                    glutin.fatal_error = Some(error);
                    true
                }
                None => {
                    glutin.fatal_error = Some(crate::Error::UnsupportedConfiguration(
                        "immediate viewport creation requires an active event loop".to_owned(),
                    ));
                    true
                }
            }
        }
    };

    if creation_failed {
        let glutin_state = glutin.borrow();
        let input = egui::RawInput {
            viewport_id,
            viewports: glutin_state
                .viewports
                .iter()
                .map(|(id, viewport)| (*id, viewport.info.clone()))
                .collect(),
            max_texture_side: glutin_state.max_texture_side,
            focused: false,
            ..Default::default()
        };
        drop(glutin_state);
        let _ = egui_ctx.run_ui(input, |ui| viewport_ui_cb(ui));
        let restore_result = restore_immediate_parent(
            &mut glutin.borrow_mut(),
            &painter.borrow(),
            ids.parent,
            previous_framebuffers,
        );
        if let Err(error) = restore_result {
            let mut glutin = glutin.borrow_mut();
            if glutin.fatal_error.is_none() {
                glutin.fatal_error = Some(error);
            }
        }
        return;
    }

    let input = {
        let mut glutin = glutin.borrow_mut();

        let Some(viewport) = glutin.viewports.get_mut(&viewport_id) else {
            return;
        };
        let (Some(egui_winit), Some(window)) = (&mut viewport.egui_winit, &viewport.window) else {
            return;
        };
        egui_winit::update_viewport_info(&mut viewport.info, egui_ctx, window, false);

        let mut raw_input = egui_winit.take_egui_input(window);
        raw_input.viewports = glutin
            .viewports
            .iter()
            .map(|(id, viewport)| (*id, viewport.info.clone()))
            .collect();
        raw_input.time = Some(beginning.elapsed().as_secs_f64());
        raw_input
    };

    // ---------------------------------------------------
    // Call the user ui-code, which could re-entrantly call this function again!
    // No locks may be hold while calling this function.

    let egui::FullOutput {
        platform_output,
        textures_delta,
        shapes,
        pixels_per_point,
        viewport_output,
    } = egui_ctx.run_ui(input, |ui| {
        viewport_ui_cb(ui);
    });

    // ---------------------------------------------------

    let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);

    let mut glutin = glutin.borrow_mut();

    let GlutinWindowContext {
        current_gl_context,
        not_current_gl_context,
        viewports,
        fatal_error,
        ..
    } = &mut *glutin;

    let Some(viewport) = viewports.get_mut(&viewport_id) else {
        if let Err(error) = restore_immediate_parent(
            &mut glutin,
            &painter.borrow(),
            ids.parent,
            previous_framebuffers,
        ) && glutin.fatal_error.is_none()
        {
            glutin.fatal_error = Some(error);
        }
        return;
    };

    viewport.info.events.clear(); // they should have been processed

    let (Some(egui_winit), Some(window), Some(gl_surface)) = (
        &mut viewport.egui_winit,
        &viewport.window,
        &viewport.gl_surface,
    ) else {
        if let Err(error) = restore_immediate_parent(
            &mut glutin,
            &painter.borrow(),
            ids.parent,
            previous_framebuffers,
        ) && glutin.fatal_error.is_none()
        {
            glutin.fatal_error = Some(error);
        }
        return;
    };

    let screen_size_in_pixels: [u32; 2] = window.inner_size().into();

    if let Err(error) = change_gl_context(current_gl_context, not_current_gl_context, gl_surface) {
        if fatal_error.is_none() {
            *fatal_error = Some(error);
        }
        previous_framebuffers.restore(painter.borrow().gl());
        return;
    }

    let current_gl_context = current_gl_context.as_ref().unwrap();

    if !gl_surface.is_current(current_gl_context) {
        log::error!(
            "egui::show_viewport_immediate: viewport {:?} ({:?}) was not created on main thread.",
            viewport.ids.this,
            viewport.builder.title
        );
    }

    unsafe {
        painter
            .borrow()
            .gl()
            .bind_framebuffer(glow::FRAMEBUFFER, None);
    }

    egui_glow::painter::clear(
        painter.borrow().gl(),
        screen_size_in_pixels,
        [0.0, 0.0, 0.0, 0.0],
    );

    painter.borrow_mut().paint_and_update_textures(
        screen_size_in_pixels,
        pixels_per_point,
        &clipped_primitives,
        &textures_delta,
    );

    {
        profiling::scope!("swap_buffers");
        if let Err(err) = gl_surface.swap_buffers(current_gl_context) {
            log::error!("swap_buffers failed: {err}");
        }
    }

    if !viewport.has_presented {
        viewport.has_presented = true;
        window.set_visible(viewport.requested_visible);
        if viewport.requested_visible && (viewport.requested_active || viewport.pending_focus) {
            window.focus_window();
            viewport.pending_focus = false;
        } else if viewport.requested_active {
            viewport.pending_focus = true;
        }
    }

    egui_winit.handle_platform_output(window, platform_output);

    let output_result = event_loop_context::with_current_event_loop(|event_loop| {
        glutin.handle_viewport_output(event_loop, egui_ctx, &viewport_output)
    });
    match output_result {
        Some(Ok(())) => {}
        Some(Err(error)) => {
            if glutin.fatal_error.is_none() {
                glutin.fatal_error = Some(error);
            }
        }
        None => {
            if glutin.fatal_error.is_none() {
                glutin.fatal_error = Some(crate::Error::UnsupportedConfiguration(
                    "immediate viewport output requires an active event loop".to_owned(),
                ));
            }
        }
    }

    if let Err(error) = restore_immediate_parent(
        &mut glutin,
        &painter.borrow(),
        ids.parent,
        previous_framebuffers,
    ) && glutin.fatal_error.is_none()
    {
        glutin.fatal_error = Some(error);
    }
}

#[cfg(feature = "__screenshot")]
fn save_screenshot_and_exit(
    path: &str,
    painter: &egui_glow::Painter,
    screen_size_in_pixels: [u32; 2],
) {
    assert!(
        path.ends_with(".png"),
        "Expected EFRAME_SCREENSHOT_TO to end with '.png', got {path:?}"
    );
    let screenshot = painter.read_screen_rgba(screen_size_in_pixels);
    image::save_buffer(
        path,
        screenshot.as_raw(),
        screenshot.width() as u32,
        screenshot.height() as u32,
        image::ColorType::Rgba8,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to save screenshot to {path:?}: {err}");
    });
    log::info!("Screenshot saved to {path:?}.");

    #[expect(clippy::exit)]
    std::process::exit(0);
}
