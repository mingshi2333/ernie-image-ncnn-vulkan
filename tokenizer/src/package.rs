// SPDX-License-Identifier: MIT
//! Full file-integrity check before inference. This detects incomplete or
//! corrupted packages; a locally supplied manifest is not a digital signature.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::{BTreeMap, BTreeSet}, fs, io::Read, path::{Component, Path}};

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn hash(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 { break; }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn safe_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.contains('\\') || name.split('/').any(|v| v.is_empty() || v == "." || v == "..")
        || Path::new(name).components().any(|v| !matches!(v, Component::Normal(_))) {
        return Err(format!("Unsafe package path: {name}"));
    }
    Ok(())
}

fn digests(value: &Value) -> Result<BTreeMap<String, String>, String> {
    let mut result = BTreeMap::new();
    for (name, digest) in value.as_object().ok_or("Missing file checksums")? {
        safe_name(name)?;
        let digest = digest.as_str().ok_or("Invalid SHA256 type")?;
        if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
            return Err(format!("Invalid SHA256: {name}"));
        }
        result.insert(name.clone(), digest.to_string());
    }
    Ok(result)
}

fn required_files() -> BTreeSet<String> {
    let mut names: BTreeSet<String> = ["model.cfg", "text/embeddings.bf16", "text/rope-inv-freq.f32",
        "dit/rope-inv-freq.f32", "vae/bn-mean.f32", "vae/bn-variance.f32",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "vae/head.ncnn.param", "vae/head.ncnn.bin"]
        .iter().map(|v| v.to_string()).collect();
    for (kind, count, stem) in [("text", 25, "text"), ("dit", 36, "block")] {
        for index in 0..count {
            for suffix in ["param", "bin"] {
                names.insert(format!("{kind}/block-{index:02}/{stem}.ncnn.{suffix}"));
            }
        }
    }
    for head in ["input", "output"] {
        for suffix in ["param", "bin"] { names.insert(format!("dit/{head}/head.ncnn.{suffix}")); }
    }
    names
}

pub fn verify(root: &Path) -> Result<(), String> {
    let manifest = read_json(&root.join("manifest.json"))?;
    let lock: Value = serde_json::from_str(include_str!("../../sources.lock.json")).map_err(|e| e.to_string())?;
    if manifest["kind"].as_str() == Some("prompt_enhancer") {
        return verify_pe(root, &manifest, &lock);
    }
    if manifest["official_model_revision"] != lock["official_model"]["revision"] {
        return Err("Official model revision differs".into());
    }
    let mut expected = digests(&manifest["files"])?;
    let schema = manifest["schema_version"].as_u64().ok_or("Missing package schema")?;
    let required = required_files();
    if schema == 1 {
        for (name, checksum) in digests(&manifest["source_manifests"])? {
            let mut path = root.join(&name);
            if path.is_dir() {
                path = path.join("model.json");
                for (filename, digest) in digests(&read_json(&path)?["files"])? {
                    expected.insert(format!("{name}/{filename}"), digest);
                }
            } else { expected.insert(name.clone(), checksum.clone()); }
            if hash(&path)? != checksum { return Err(format!("Source manifest checksum differs: {name}")); }
        }
        let tokenizer = read_json(&root.join("tokenizer/manifest.json"))?;
        if tokenizer["revision"] != manifest["official_model_revision"] { return Err("Tokenizer revision differs".into()); }
        for (name, checksum) in digests(&tokenizer["files"])? {
            let key = format!("tokenizer/{name}");
            if expected.get(&key).is_some_and(|v| v != &checksum) { return Err("Tokenizer manifests disagree".into()); }
            expected.insert(key, checksum);
        }
    } else if schema == 2 {
        if manifest["ncnn_revision"] != lock["ncnn"]["revision"] { return Err("ncnn revision differs".into()); }
        let sizes = manifest["file_sizes"].as_object().ok_or("Missing file sizes")?;
        if expected.keys().cloned().collect::<BTreeSet<_>>() != required
            || sizes.keys().cloned().collect::<BTreeSet<_>>() != required {
            return Err("Runtime file inventory differs".into());
        }
        if !manifest["portable"].is_boolean() { return Err("Missing portable package flag".into()); }
    } else { return Err("Unsupported package schema".into()); }

    for name in required {
        let path = root.join(&name);
        let checksum = expected.get(&name).ok_or_else(|| format!("Missing runtime checksum: {name}"))?;
        if manifest["portable"].as_bool() == Some(true) {
            let mut component = root.to_path_buf();
            for part in Path::new(&name).components() {
                component.push(part);
                if component.is_symlink() { return Err(format!("Portable package contains a symlink: {name}")); }
            }
        }
        let metadata = fs::metadata(&path).map_err(|e| format!("{name}: {e}"))?;
        if !metadata.is_file() { return Err(format!("Not a runtime file: {name}")); }
        if schema == 2 && manifest["file_sizes"][&name].as_u64() != Some(metadata.len()) {
            return Err(format!("File size differs: {name}"));
        }
        if hash(&path)? != *checksum { return Err(format!("File checksum differs: {name}")); }
    }
    let config = fs::read_to_string(root.join("model.cfg")).map_err(|e| e.to_string())?;
    let words: Vec<_> = config.split_whitespace().collect();
    if words.len() != 12 { return Err("Malformed model.cfg".into()); }
    let mut values = serde_json::Map::new();
    for pair in words.chunks_exact(2) {
        let number: i64 = pair[1].parse().map_err(|_| "Invalid model.cfg integer")?;
        if values.insert(pair[0].to_string(), number.into()).is_some() { return Err("Duplicate model.cfg entry".into()); }
    }
    if Value::Object(values) != manifest["config"] { return Err("Configuration and manifest disagree".into()); }
    Ok(())
}

fn verify_pe(root: &Path, manifest: &Value, lock: &Value) -> Result<(), String> {
    if manifest["schema_version"].as_u64() != Some(1)
        || manifest["official_model_revision"] != lock["official_model"]["revision"]
        || manifest["ncnn_revision"] != lock["ncnn"]["revision"] {
        return Err("PE package version differs".into());
    }
    if !manifest["portable"].is_boolean() { return Err("Missing PE portable flag".into()); }
    let mut required: BTreeSet<String> = ["pe.cfg", "embeddings.bf16", "rope-inv-freq.f32",
        "head.ncnn.param", "head.ncnn.bin", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
        "tokenizer/chat_template.jinja"].iter().map(|s| s.to_string()).collect();
    for i in 0..26 {
        for suffix in ["param", "bin"] { required.insert(format!("block-{i:02}/pe.ncnn.{suffix}")); }
    }
    let files = digests(&manifest["files"])?;
    let sizes = manifest["file_sizes"].as_object().ok_or("Missing PE file sizes")?;
    if files.keys().cloned().collect::<BTreeSet<_>>() != required
        || sizes.keys().cloned().collect::<BTreeSet<_>>() != required {
        return Err("PE runtime file inventory differs".into());
    }
    for name in required {
        let path = root.join(&name);
        if manifest["portable"].as_bool() == Some(true) {
            let mut component = root.to_path_buf();
            for part in Path::new(&name).components() {
                component.push(part);
                if component.is_symlink() { return Err(format!("Portable PE package contains a symlink: {name}")); }
            }
        }
        let metadata = fs::metadata(&path).map_err(|e| format!("{name}: {e}"))?;
        if !metadata.is_file() || sizes[&name].as_u64() != Some(metadata.len()) {
            return Err(format!("PE file size differs: {name}"));
        }
        if hash(&path)? != files[&name] { return Err(format!("PE file checksum differs: {name}")); }
    }
    if hash(&root.join("tokenizer/chat_template.jinja"))? != "0c859484eecf01db103acd02c332610163ee425cd46866d6cb126ee1bee974ea" {
        return Err("PE chat template differs from native formatter".into());
    }
    let expected = serde_json::json!({"layers":26,"hidden_size":3072,"vocabulary":131072,"capacity":4096,"tokens_per_call":1});
    if manifest["config"] != expected { return Err("PE configuration differs".into()); }
    let config = fs::read_to_string(root.join("pe.cfg")).map_err(|e| e.to_string())?;
    let words: Vec<_> = config.split_whitespace().collect();
    if words.len() != 10 { return Err("Malformed pe.cfg".into()); }
    let mut parsed = serde_json::Map::new();
    for pair in words.chunks_exact(2) {
        let value: u64 = pair[1].parse().map_err(|_| "Invalid PE configuration integer")?;
        if parsed.insert(pair[0].to_string(), value.into()).is_some() { return Err("Duplicate PE configuration field".into()); }
    }
    if Value::Object(parsed) != expected { return Err("PE configuration and manifest disagree".into()); }
    Ok(())
}
