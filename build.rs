//! Generates Rust bindings for the KiCad IPC API from a pre-extracted FileDescriptorSet
//! (see tools/extract_kicad_proto.py). Using a descriptor set means no `protoc` binary and no
//! KiCad source checkout are needed at build time.

use prost::Message;

fn main() {
    let fds_path = "proto/kicad_api.fds.bin";
    println!("cargo:rerun-if-changed={fds_path}");
    let bytes = std::fs::read(fds_path).expect("read proto/kicad_api.fds.bin");
    let fds = prost_types::FileDescriptorSet::decode(bytes.as_slice()).expect("decode descriptor set");

    let mut cfg = prost_build::Config::new();
    cfg.include_file("kiapi_includes.rs");
    // `prost::Name` impls give each message its `type.googleapis.com/...` Any type URL.
    cfg.enable_type_names();
    cfg.type_name_domain(["."], "type.googleapis.com");
    cfg.compile_fds(fds).expect("prost-build compile_fds");
}
