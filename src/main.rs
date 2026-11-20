//! Tracewright KiCad plugin entry point. KiCad launches this executable from the PCB editor's
//! toolbar (IPC API plugin, runtime "exec"); it reads the open board over the API.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> eframe::Result {
    tracewright::ui::run()
}
