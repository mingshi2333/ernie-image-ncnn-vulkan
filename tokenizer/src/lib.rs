// SPDX-License-Identifier: MIT
//! Narrow C ABI around the official Tokenizers implementation. The C++ owner
//! supplies valid buffers and owns the handle from create until destroy.
use std::{panic::{catch_unwind, AssertUnwindSafe}, path::Path, slice, str};
use tokenizers::{Tokenizer, TruncationParams};
mod package;

#[no_mangle]
pub unsafe extern "C" fn ernie_package_verify(directory: *const u8, length: usize,
    error: *mut u8, error_capacity: usize) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| package::verify(Path::new(text(directory, length)?)))) {
        Ok(Ok(())) => 0,
        Ok(Err(message)) => { write_error(error, error_capacity, &message); -1 },
        Err(_) => { write_error(error, error_capacity, "Package verification panicked"); -1 }
    }
}

pub struct Handle { tokenizer: Tokenizer, maximum: usize, bos: u32 }

unsafe fn write_error(destination: *mut u8, capacity: usize, message: &str) {
    if destination.is_null() || capacity == 0 { return; }
    let length = message.len().min(capacity - 1);
    std::ptr::copy_nonoverlapping(message.as_ptr(), destination, length);
    *destination.add(length) = 0;
}

unsafe fn text<'a>(bytes: *const u8, length: usize) -> Result<&'a str, String> {
    if length == 0 { return Ok(""); }
    if bytes.is_null() { return Err("Null text buffer".into()); }
    str::from_utf8(slice::from_raw_parts(bytes, length)).map_err(|e| e.to_string())
}

#[no_mangle]
pub unsafe extern "C" fn ernie_tok_create(directory: *const u8, length: usize,
    error: *mut u8, error_capacity: usize) -> *mut Handle {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Handle, String> {
        let path = Path::new(text(directory, length)?);
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("tokenizer_config.json")).map_err(|e| e.to_string())?
        ).map_err(|e| e.to_string())?;
        let maximum = config["model_max_length"].as_u64().ok_or("Missing model_max_length")? as usize;
        if maximum != 2048 || config["tokenizer_class"].as_str() != Some("TokenizersBackend") {
            return Err("Tokenizer configuration differs from the reviewed ERNIE contract".into());
        }
        let bos_token = config["bos_token"].as_str().ok_or("Missing BOS token")?;
        let mut tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(|e| e.to_string())?;
        let bos = tokenizer.token_to_id(bos_token).ok_or("BOS token absent from vocabulary")?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(Some(TruncationParams { max_length: maximum, ..Default::default() }))
            .map_err(|e| e.to_string())?;
        Ok(Handle { tokenizer, maximum, bos })
    }));
    match result {
        Ok(Ok(handle)) => Box::into_raw(Box::new(handle)),
        Ok(Err(message)) => { write_error(error, error_capacity, &message); std::ptr::null_mut() },
        Err(_) => { write_error(error, error_capacity, "Tokenizer initialization panicked"); std::ptr::null_mut() }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ernie_tok_maximum(handle: *const Handle) -> usize {
    handle.as_ref().map_or(0, |h| h.maximum)
}

#[no_mangle]
pub unsafe extern "C" fn ernie_tok_encode(handle: *const Handle, prompt: *const u8, length: usize,
    output: *mut u32, capacity: usize, written: *mut usize, error: *mut u8, error_capacity: usize) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let handle = handle.as_ref().ok_or("Null tokenizer handle")?;
        if output.is_null() || written.is_null() { return Err("Null output buffer".into()); }
        let encoding = handle.tokenizer.encode(text(prompt, length)?, true).map_err(|e| e.to_string())?;
        let ids = encoding.get_ids();
        let fallback = [handle.bos];
        let ids = if ids.is_empty() { &fallback[..] } else { ids };
        *written = ids.len();
        if ids.len() > capacity { return Err("Insufficient token output capacity".into()); }
        std::ptr::copy_nonoverlapping(ids.as_ptr(), output, ids.len());
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(message)) => { write_error(error, error_capacity, &message); -1 },
        Err(_) => { write_error(error, error_capacity, "Tokenizer encoding panicked"); -1 }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ernie_tok_destroy(handle: *mut Handle) {
    if !handle.is_null() { drop(Box::from_raw(handle)); }
}
