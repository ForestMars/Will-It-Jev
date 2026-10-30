fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true) // test sidecars in src/pipeline/tier.rs
        .build_client(true)
        .compile_protos(&["proto/tier3.proto"], &["proto"])?;
    Ok(())
}