//! Native display/configuration creation for a Glow controller without a presentation window.

use glutin::config::ConfigTemplateBuilder;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use glutin::display::GlDisplay as _;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
use winit::{event_loop::ActiveEventLoop, window::Window};

/// Creates a display and chooses a config, retrying opaque configs on capability mismatch.
///
/// On Windows, the returned private, hidden bootstrap window must outlive every GL object using
/// the config: glutin's WGL config retains its HDC, including for later child-surface queries.
/// It is never registered as an egui viewport or used for presentation or repaint scheduling.
/// macOS creates a CGL display without a native window. Unsupported platforms return an error.
#[expect(unsafe_code)]
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(super) fn create_display(
    egui_ctx: &egui::Context,
    event_loop: &ActiveEventLoop,
    template: ConfigTemplateBuilder,
) -> crate::Result<(Option<Window>, glutin::config::Config)> {
    #[cfg(target_os = "windows")]
    let window = Some(egui_winit::create_window(
        egui_ctx,
        event_loop,
        &egui::ViewportBuilder::default()
            .with_title("eframe GL resources")
            .with_inner_size([1.0, 1.0])
            .with_visible(false)
            .with_active(false)
            .with_decorations(false)
            .with_taskbar(false),
    )?);
    #[cfg(not(target_os = "windows"))]
    let window: Option<Window> = None;
    let _ = egui_ctx;

    let raw_window = window
        .as_ref()
        .map(|window| window.window_handle().map(|handle| handle.as_raw()))
        .transpose()
        .map_err(|err| crate::Error::UnsupportedConfiguration(err.to_string()))?;
    let template = if let Some(handle) = raw_window {
        template.compatible_with_native_window(handle)
    } else {
        template
    };
    // The same config must support presentation windows and the persistent controller pbuffer.
    #[cfg(target_os = "windows")]
    let template = template.with_surface_type(
        glutin::config::ConfigSurfaceTypes::WINDOW | glutin::config::ConfigSurfaceTypes::PBUFFER,
    );

    #[cfg(target_os = "macos")]
    let preference = glutin::display::DisplayApiPreference::Cgl;
    #[cfg(target_os = "windows")]
    let preference = glutin::display::DisplayApiPreference::WglThenEgl(raw_window);
    let raw_display = event_loop
        .display_handle()
        .map_err(|err| crate::Error::UnsupportedConfiguration(err.to_string()))?;
    // SAFETY: the event loop and private bootstrap remain alive through display/config use.
    let display = unsafe { glutin::display::Display::new(raw_display.as_raw(), preference)? };
    let config = select_config(|transparent| {
        let template = template.clone().with_transparency(transparent).build();
        // SAFETY: any native window in this template is owned by `window` above.
        unsafe {
            display
                .find_configs(template)
                .map(|mut configs| configs.next())
        }
    })?;
    Ok((window, config))
}

/// Rejects platforms that do not have a supported windowless GL controller target.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub(super) fn create_display(
    _: &egui::Context,
    _: &ActiveEventLoop,
    _: ConfigTemplateBuilder,
) -> crate::Result<(Option<Window>, glutin::config::Config)> {
    Err(crate::Error::UnsupportedConfiguration(
        "windowless Glow requires macOS or Windows".to_owned(),
    ))
}

/// Retries an opaque query when a transparency request fails or returns no matching configs.
/// Both absence and errors are ordinary capability failures; neither may panic in a picker.
#[cfg(any(test, target_os = "macos", target_os = "windows"))]
fn select_config<T>(
    mut query: impl FnMut(bool) -> Result<Option<T>, glutin::error::Error>,
) -> crate::Result<T> {
    match query(true) {
        Ok(Some(config)) => return Ok(config),
        Ok(None) => log::debug!("No transparent GL config; trying opaque configs"),
        Err(err) => log::debug!("Transparent GL config query failed: {err}; trying opaque configs"),
    }
    query(false)?.ok_or_else(|| {
        crate::Error::UnsupportedConfiguration(
            "no GL config supports the requested windowless controller surfaces".to_owned(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_transparent_query_retries_opaque() {
        let mut requests = Vec::new();
        let selected = select_config(|transparent| {
            requests.push(transparent);
            Ok((!transparent).then_some(7))
        })
        .unwrap();
        assert_eq!(selected, 7, "must return the opaque config");
        assert_eq!(
            requests,
            [true, false],
            "must try transparency before opaque"
        );
    }

    #[test]
    fn failed_transparent_query_retries_opaque() {
        let selected = select_config(|transparent| {
            if transparent {
                Err(glutin::error::ErrorKind::BadConfig.into())
            } else {
                Ok(Some(7))
            }
        })
        .unwrap();
        assert_eq!(
            selected, 7,
            "a failed transparency query must still allow an opaque config"
        );
    }

    #[test]
    fn absent_configs_return_an_error() {
        assert!(
            select_config::<()>(|_| Ok(None)).is_err(),
            "no configs must be an error"
        );
    }
}
