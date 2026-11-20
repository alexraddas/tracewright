//! Tracewright: GPU-accelerated orthogonal PCB autorouter for KiCad.
pub mod board;
pub mod drc;
pub mod geom;
pub mod grid;
pub mod kicad;
pub mod render;
pub mod router;
pub mod search;
pub mod ui;
pub mod settings;

pub mod proto {
    #![allow(clippy::all, non_camel_case_types, dead_code)]
    include!(concat!(env!("OUT_DIR"), "/kiapi_includes.rs"));
}
