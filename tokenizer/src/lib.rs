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

fn create_tokenizer(tokenizer_path: &Path, config_path: &Path) -> Result<Handle,String> {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(config_path).map_err(|e| e.to_string())?
        ).map_err(|e| e.to_string())?;
        let maximum = config["model_max_length"].as_u64().ok_or("Missing model_max_length")? as usize;
        if maximum != 2048 || config["tokenizer_class"].as_str() != Some("TokenizersBackend") {
            return Err("Tokenizer configuration differs from the reviewed ERNIE contract".into());
        }
        let bos_token = config["bos_token"].as_str().ok_or("Missing BOS token")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
        let bos = tokenizer.token_to_id(bos_token).ok_or("BOS token absent from vocabulary")?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(Some(TruncationParams { max_length: maximum, ..Default::default() }))
            .map_err(|e| e.to_string())?;
        Ok(Handle { tokenizer, maximum, bos })
}
#[no_mangle]
pub unsafe extern "C" fn ernie_tok_create(directory:*const u8,length:usize,error:*mut u8,error_capacity:usize)->*mut Handle {
    finish_tokenizer(catch_unwind(AssertUnwindSafe(|| {
        let path=Path::new(text(directory,length)?);
        create_tokenizer(&path.join("tokenizer.json"),&path.join("tokenizer_config.json"))
    })),error,error_capacity)
}
#[no_mangle]
pub unsafe extern "C" fn ernie_tok_create_files(tokenizer:*const u8,tokenizer_length:usize,config:*const u8,config_length:usize,error:*mut u8,error_capacity:usize)->*mut Handle {
    finish_tokenizer(catch_unwind(AssertUnwindSafe(||create_tokenizer(Path::new(text(tokenizer,tokenizer_length)?),Path::new(text(config,config_length)?)))),error,error_capacity)
}
unsafe fn finish_tokenizer(result:std::thread::Result<Result<Handle,String>>,error:*mut u8,error_capacity:usize)->*mut Handle {
    match result {
        Ok(Ok(handle))=>Box::into_raw(Box::new(handle)),
        Ok(Err(message))=>{write_error(error,error_capacity,&message);std::ptr::null_mut()},
        Err(_)=>{write_error(error,error_capacity,"Tokenizer initialization panicked");std::ptr::null_mut()}
    }
}

#[no_mangle]
pub unsafe extern "C" fn ernie_tok_maximum(handle: *const Handle) -> usize {
    handle.as_ref().map_or(0, |h| h.maximum)
}

#[no_mangle]
pub unsafe extern "C" fn ernie_tok_encode(handle: *const Handle, prompt: *const u8, length: usize,
    output: *mut u32, capacity: usize, written: *mut usize, error: *mut u8, error_capacity: usize) -> i32 {
    encode(handle, prompt, length, output, capacity, written, error, error_capacity, true)
}

#[no_mangle]
pub unsafe extern "C" fn ernie_tok_encode_exact(handle: *const Handle, prompt: *const u8, length: usize,
    output: *mut u32, capacity: usize, written: *mut usize, error: *mut u8, error_capacity: usize) -> i32 {
    encode(handle, prompt, length, output, capacity, written, error, error_capacity, false)
}

unsafe fn encode(handle: *const Handle, prompt: *const u8, length: usize,
    output: *mut u32, capacity: usize, written: *mut usize, error: *mut u8, error_capacity: usize, truncate: bool) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let handle = handle.as_ref().ok_or("Null tokenizer handle")?;
        if output.is_null() || written.is_null() { return Err("Null output buffer".into()); }
        let mut untruncated;
        let tokenizer = if truncate { &handle.tokenizer } else {
            untruncated = handle.tokenizer.clone();
            untruncated.with_truncation(None).map_err(|e| e.to_string())?;
            &untruncated
        };
        let encoding = tokenizer.encode(text(prompt, length)?, true).map_err(|e| e.to_string())?;
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

unsafe fn string_result(result: Result<String, String>, output: *mut u8, capacity: usize,
    written: *mut usize, error: *mut u8, error_capacity: usize) -> i32 {
    match result {
        Ok(value) => {
            if written.is_null() { write_error(error, error_capacity, "Null output length"); return -1; }
            *written = value.len();
            if output.is_null() && capacity == 0 { return 0; }
            if output.is_null() || capacity < value.len() {
                write_error(error, error_capacity, "Insufficient text capacity"); return -1;
            }
            std::ptr::copy_nonoverlapping(value.as_ptr(), output, value.len());
            0
        },
        Err(message) => { write_error(error, error_capacity, &message); -1 }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ernie_tok_decode(handle: *const Handle, ids: *const u32, length: usize,
    output: *mut u8, capacity: usize, written: *mut usize, error: *mut u8, error_capacity: usize) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        let handle = handle.as_ref().ok_or("Null tokenizer handle")?;
        let ids = if length == 0 { &[][..] } else {
            if ids.is_null() { return Err("Null token IDs".into()); }
            slice::from_raw_parts(ids, length)
        };
        handle.tokenizer.decode(ids, true).map_err(|e| e.to_string())
    })).unwrap_or_else(|_| Err("Tokenizer decoding panicked".into()));
    string_result(result, output, capacity, written, error, error_capacity)
}

#[no_mangle]
pub unsafe extern "C" fn ernie_pe_prompt(prompt: *const u8, length: usize, width: u32, height: u32,
    output: *mut u8, capacity: usize, written: *mut usize, error: *mut u8, error_capacity: usize) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
            return Err("Invalid PE image dimensions".into());
        }
        // Exact pinned chat template and Python json.dumps(ensure_ascii=False)
        // separators; the PE tokenizer itself does not add BOS.
        let quoted = serde_json::to_string(text(prompt, length)?).map_err(|e| e.to_string())?;
        Ok(format!("<s>[SYSTEM_PROMPT]你是一个专业的文生图 Prompt 增强助手。你将收到用户的简短图片描述及目标生成分辨率，请据此扩写为一段内容丰富、细节充分的视觉描述，以帮助文生图模型生成高质量的图片。仅输出增强后的描述，不要包含任何解释或前缀。[/SYSTEM_PROMPT][INST]{{\"prompt\": {quoted}, \"width\": {width}, \"height\": {height}}}[/INST]"))
    })).unwrap_or_else(|_| Err("PE prompt formatting panicked".into()));
    string_result(result, output, capacity, written, error, error_capacity)
}

#[no_mangle]
pub unsafe extern "C" fn ernie_trim_text(input: *const u8, length: usize,
    output: *mut u8, capacity: usize, written: *mut usize, error: *mut u8, error_capacity: usize) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<String, String> {
        Ok(text(input, length)?.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)).to_string())
    })).unwrap_or_else(|_| Err("Text trimming panicked".into()));
    string_result(result, output, capacity, written, error, error_capacity)
}

#[no_mangle]
pub unsafe extern "C" fn ernie_tok_destroy(handle: *mut Handle) {
    if !handle.is_null() { drop(Box::from_raw(handle)); }
}

// Internal bounded graph-text hashing for the independent shape-contract tool.
// This does not enable a new runtime package schema.
#[no_mangle]
pub unsafe extern "C" fn ernie_shape_sha256(bytes: *const u8, length: usize, output: *mut u8) -> i32 {
    use sha2::{Digest, Sha256};
    if output.is_null() || (length != 0 && bytes.is_null()) || length > 1024 * 1024 { return -1; }
    match catch_unwind(AssertUnwindSafe(|| {
        let data = if length == 0 { &[] } else { slice::from_raw_parts(bytes, length) };
        let digest = Sha256::digest(data);
        std::ptr::copy_nonoverlapping(digest.as_ptr(), output, 32);
    })) { Ok(()) => 0, Err(_) => -1 }
}

mod shared_package;
#[no_mangle]
pub unsafe extern "C" fn ernie_model_package_open(directory:*const u8,length:usize,width:i32,height:i32,error:*mut u8,capacity:usize)->*mut shared_package::ResolvedPackage{
    match catch_unwind(AssertUnwindSafe(||shared_package::open(Path::new(text(directory,length)?),width,height))){
        Ok(Ok(p))=>Box::into_raw(Box::new(p)),
        Ok(Err(e))=>{write_error(error,capacity,&e);std::ptr::null_mut()},
        Err(_)=>{write_error(error,capacity,"Package open panicked");std::ptr::null_mut()}
    }
}
#[no_mangle]
pub unsafe extern "C" fn ernie_model_package_destroy(handle:*mut shared_package::ResolvedPackage){if !handle.is_null(){drop(Box::from_raw(handle));}}
#[no_mangle]
pub unsafe extern "C" fn ernie_model_package_config(handle:*const shared_package::ResolvedPackage,config:*mut i32)->i32{
    match handle.as_ref(){Some(p) if !config.is_null()=>{std::ptr::copy_nonoverlapping(p.config.as_ptr(),config,4);p.schema as i32},_=>-1}
}
#[no_mangle]
pub unsafe extern "C" fn ernie_model_package_file(handle:*const shared_package::ResolvedPackage,name:*const u8,length:usize,output:*mut u8,capacity:usize)->i32{
    let result=catch_unwind(AssertUnwindSafe(||->Result<(),String>{
        let p=handle.as_ref().ok_or("Null package handle")?;
        let path=p.files.get(text(name,length)?).ok_or("Unknown logical package file")?;
        let value=path.to_str().ok_or("NonUTF8 object path")?;
        if output.is_null()||value.len()>=capacity{return Err("Object path buffer too small".into());}
        write_error(output,capacity,value);Ok(())
    }));if matches!(result,Ok(Ok(()))){0}else{-1}
}
