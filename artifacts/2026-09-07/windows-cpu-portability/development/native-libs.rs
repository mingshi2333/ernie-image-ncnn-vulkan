#[no_mangle]
pub extern "C" fn ernie_rust_dependency_probe() -> usize { std::env::vars_os().count() }
