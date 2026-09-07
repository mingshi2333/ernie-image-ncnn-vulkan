// SPDX-License-Identifier: MIT
// Strict recursive JSON parsing rejects duplicate metadata keys before validation.
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use std::fmt;
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D:Deserializer<'de>>(d:D)->Result<Self,D::Error>{
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value=Unique;
            fn expecting(&self,f:&mut fmt::Formatter)->fmt::Result{f.write_str("JSON without duplicate keys")}
            fn visit_bool<E:de::Error>(self,v:bool)->Result<Unique,E>{Ok(Unique(v.into()))}
            fn visit_i64<E:de::Error>(self,v:i64)->Result<Unique,E>{Ok(Unique(v.into()))}
            fn visit_u64<E:de::Error>(self,v:u64)->Result<Unique,E>{Ok(Unique(v.into()))}
            fn visit_f64<E:de::Error>(self,v:f64)->Result<Unique,E>{serde_json::Number::from_f64(v).map(|n|Unique(Value::Number(n))).ok_or_else(||E::custom("Nonfinite JSON"))}
            fn visit_str<E:de::Error>(self,v:&str)->Result<Unique,E>{Ok(Unique(v.into()))}
            fn visit_string<E:de::Error>(self,v:String)->Result<Unique,E>{Ok(Unique(v.into()))}
            fn visit_unit<E:de::Error>(self)->Result<Unique,E>{Ok(Unique(Value::Null))}
            fn visit_none<E:de::Error>(self)->Result<Unique,E>{self.visit_unit()}
            fn visit_seq<A:SeqAccess<'de>>(self,mut a:A)->Result<Unique,A::Error>{let mut out=Vec::new();while let Some(Unique(v))=a.next_element()?{out.push(v);}Ok(Unique(Value::Array(out)))}
            fn visit_map<A:MapAccess<'de>>(self,mut a:A)->Result<Unique,A::Error>{let mut out=serde_json::Map::new();while let Some((k,Unique(v)))=a.next_entry::<String,Unique>()?{if out.insert(k,v).is_some(){return Err(de::Error::custom("Duplicate JSON key"));}}Ok(Unique(Value::Object(out)))}
        }
        d.deserialize_any(V)
    }
}
fn strict_json(bytes:&[u8])->Result<Value,String>{serde_json::from_slice::<Unique>(bytes).map(|v|v.0).map_err(|e|e.to_string())}

// Shared immutable objects, restricted to pinned independently exported static sources.
use crate::package::{hash, required_files};
use serde_json::Value;
use std::{collections::{BTreeMap, BTreeSet}, fs, path::{Path, PathBuf}};

#[derive(Clone)]
pub struct PackageSource {
    pub config: [i32; 4],
    pub files: BTreeMap<String, PathBuf>,
    source_manifest: String,
}
pub struct ResolvedPackage {
    pub config: [i32; 4],
    pub source_config: [i32; 4],
    pub files: BTreeMap<String, PathBuf>,
    pub schema: u32,
    // Verified path/metadata records only. Selection never reloads or rehashes
    // the weight store and never changes an independently exported text graph.
    sources: Vec<PackageSource>,
}
fn json(path: &Path) -> Result<Value,String> {
    let meta=fs::symlink_metadata(path).map_err(|e|e.to_string())?;
    if !meta.is_file() || meta.len()>4*1024*1024 {return Err("Invalid bounded package metadata".into());}
    strict_json(&fs::read(path).map_err(|e|e.to_string())?)
}
fn keys(v:&Value, expected:&[&str])->Result<(),String>{
    let actual:BTreeSet<_>=v.as_object().ok_or("Expected object")?.keys().map(String::as_str).collect();
    if actual!=expected.iter().copied().collect(){return Err("Unexpected schema-3 fields".into());} Ok(())
}
fn sha(s:&str)->Result<(),String>{if s.len()!=64 || !s.bytes().all(|c|c.is_ascii_digit()||(b'a'..=b'f').contains(&c)){return Err("Invalid object digest".into());}Ok(())}
fn config(v:&Value)->Result<[i32;4],String>{
    let mut out=[0;4];for (i,k) in ["packed_width","packed_height","text_bucket","dit_text_tokens"].iter().enumerate(){out[i]=i32::try_from(v[k].as_i64().ok_or("Invalid config integer")?).map_err(|_|"Config overflow")?;}Ok(out)
}
pub fn verify(root:&Path)->Result<Vec<PackageSource>,String>{
    let contract:Value=serde_json::from_str(include_str!("../schema3_contract.json")).map_err(|e|e.to_string())?;
    verify_contract(root,&contract)
}
fn verify_contract(root:&Path,contract:&Value)->Result<Vec<PackageSource>,String>{
    let m=json(&root.join("manifest.json"))?;
    keys(&m,&["schema_version","format","instances","objects","math","encoder","generation_quality_status"])?;
    for key in ["schema_version","format","math","generation_quality_status"] {if m[key]!=contract[key]{return Err(format!("Unreviewed schema-3 {key}"));}}
    let encoder=&m["encoder"];
    let default_encoder=&contract["encoder"];
    let reviewed=contract["reviewed_encoders"].as_object().ok_or("Missing reviewed encoder registry")?;
    let mut encoder_source:Option<&str>=None;
    if encoder!=default_encoder {
        for (source,value) in reviewed {
            let mut expected=value.clone();
            let expected_object=expected.as_object_mut().ok_or("Invalid reviewed encoder")?;
            let files=expected_object.remove("files").ok_or("Missing reviewed encoder files")?;
            expected_object.remove("evidence");
            let mut bindings=serde_json::Map::new();
            for (name,item) in files.as_object().ok_or("Invalid reviewed encoder files")? {
                bindings.insert(name.clone(),item["sha256"].clone());
            }
            expected_object.insert("files".into(),Value::Object(bindings));
            if encoder==&expected {if encoder_source.replace(source).is_some(){return Err("Ambiguous reviewed encoder".into());}}
        }
        if encoder_source.is_none(){return Err("Unreviewed schema-3 encoder".into());}
    }
    let objects=m["objects"].as_object().ok_or("Missing objects")?;
    let instances=m["instances"].as_array().ok_or("Missing instances")?;
    if instances.is_empty()||instances.len()>3{return Err("Invalid static instance count".into());}
    if fs::symlink_metadata(root.join("objects")).map_err(|e|e.to_string())?.file_type().is_symlink(){return Err("Symlink object store".into());}
    let mut used=BTreeSet::new(); let mut seen=BTreeSet::new(); let mut out=Vec::new();
    let mut object=|digest:&str|->Result<PathBuf,String>{
        sha(digest)?;let size=objects.get(digest).and_then(Value::as_u64).ok_or("Missing/invalid object size")?;
        let path=root.join("objects").join(digest);
        let meta=fs::symlink_metadata(&path).map_err(|e|e.to_string())?;
        if !meta.is_file()||meta.len()!=size{return Err("Object type/size mismatch".into());}
        if used.insert(digest.to_owned()) && hash(&path)?!=digest{return Err("Object checksum mismatch".into());} Ok(path)
    };
    for instance in instances {
        keys(instance,&["source_manifest_sha256","config","runtime_bindings"])?;
        let digest=instance["source_manifest_sha256"].as_str().ok_or("Missing source digest")?;
        let pinned=contract["source_manifests"].get(digest).ok_or("Unknown source manifest")?;
        if !seen.insert(digest){return Err("Duplicate source manifest".into());}
        let source=json(&object(digest)?)?;
        if source["schema_version"].as_u64()!=Some(2)||source["portable"]!=true||source["config"]!=*pinned||instance["config"]!=*pinned||source["files"]!=instance["runtime_bindings"]{return Err("Pinned instance differs".into());}
        let bindings=source["files"].as_object().ok_or("Missing bindings")?;
        let sizes=source["file_sizes"].as_object().ok_or("Missing sizes")?;
        if bindings.keys().cloned().collect::<BTreeSet<_>>()!=required_files() || sizes.keys().cloned().collect::<BTreeSet<_>>()!=required_files(){return Err("Incomplete runtime layers/files".into());}
        let mut files=BTreeMap::new();
        for(name,digest) in bindings {
            let path=object(digest.as_str().ok_or("Invalid binding")?)?;
            if sizes[name].as_u64()!=Some(fs::metadata(&path).map_err(|e|e.to_string())?.len()){return Err("Pinned file size mismatch".into());}
            files.insert(name.clone(),path);
        }
        if encoder_source==Some(digest) {
            let trusted=&reviewed[digest];
            for(name,item) in trusted["files"].as_object().ok_or("Invalid reviewed encoder files")? {
                let expected_digest=item["sha256"].as_str().ok_or("Invalid encoder digest")?;
                let path=object(expected_digest)?;
                if item["size"].as_u64()!=Some(fs::metadata(&path).map_err(|e|e.to_string())?.len()) {return Err("Reviewed encoder size mismatch".into());}
                if let Some(existing)=files.get(name) {
                    if existing!=&path{return Err("Encoder file conflicts with instance binding".into());}
                } else { files.insert(name.clone(),path); }
            }
        }
        // Exact source manifest checksum binds model.cfg, all layer graphs and weights.
        // C++ additionally enforces the canonical full-graph allowlist before instantiation.
        let source_config = config(pinned)?;
        out.push(PackageSource{config:source_config,files,
                                 source_manifest:digest.to_owned()});
    }
    if encoder_source.is_some_and(|source|!seen.contains(source)) {return Err("Reviewed encoder source instance is absent".into());}
    if used!=objects.keys().cloned().collect(){return Err("Unbound shared objects".into());}
    let entries=fs::read_dir(root.join("objects")).map_err(|e|e.to_string())?.map(|e| e.map_err(|e|e.to_string()).and_then(|e|e.file_name().into_string().map_err(|_|"Invalid object name".into()))).collect::<Result<BTreeSet<_>,String>>()?;
    if entries!=used{return Err("Unlisted shared objects".into());}
    let entries=fs::read_dir(root).map_err(|e|e.to_string())?.map(|e|e.map_err(|e|e.to_string()).and_then(|e|e.file_name().into_string().map_err(|_|"Invalid package name".into()))).collect::<Result<BTreeSet<_>,String>>()?;
    if entries!=["manifest.json".to_string(),"objects".to_string()].into_iter().collect(){return Err("Unlisted package entries".into());}
    Ok(out)
}
fn validate_dimensions(width:i32,height:i32)->Result<(),String> {
    if width<16 || height<16 || width>2048 || height>2048 || width%16!=0 || height%16!=0 ||
        i64::from(width)*i64::from(height)>2097152 {
        return Err("Runtime dimensions require multiples of 16 in [16,2048], area at most 2097152".into());
    }
    Ok(())
}
fn source_at_shape(sources:&[PackageSource], index:usize, width:i32, height:i32)
    ->([i32;4], [i32;4], BTreeMap<String,PathBuf>)
{
    let source=&sources[index];
    let mut target=source.config;
    target[0]=width/16; target[1]=height/16;
    let mut files=source.files.clone();
    // Encoder execution has its own spatial evidence. A text-source change can
    // keep an encoder for the requested geometry, never one for another size.
    for name in ["vae/encoder.ncnn.param", "vae/encoder.ncnn.bin"] {
        files.remove(name);
        if let Some(path)=sources.iter().filter(|p|p.config[0]*16==width && p.config[1]*16==height)
            .find_map(|p|p.files.get(name)) { files.insert(name.into(),path.clone()); }
    }
    (target,source.config,files)
}
fn select(sources:Vec<PackageSource>,contract:&Value,mut width:i32,mut height:i32)
    ->Result<ResolvedPackage,String>
{
    if sources.is_empty() {return Err("Missing shared sources".into());}
    if width==0 {
        if height!=0 || sources.len()!=1 {return Err("Multiple static instances require explicit width and height".into());}
        width=sources[0].config[0]*16; height=sources[0].config[1]*16;
    }
    validate_dimensions(width,height)?;
    let mut seen=BTreeSet::new();
    for source in &sources {
        let pinned=contract["source_manifests"].get(&source.source_manifest).ok_or("Unknown source manifest")?;
        if config(pinned)?!=source.config || !seen.insert(&source.source_manifest) {
            return Err("Unknown or duplicate shared source configuration".into());
        }
    }
    // Before tokenization, prefer the original geometry. This preserves legacy
    // shared-package defaults and lets img2img use its separately reviewed encoder.
    let index=(0..sources.len()).min_by_key(|&i| {
        let c=sources[i].config;
        (c[0]*16!=width || c[1]*16!=height,c[2])
    }).unwrap();
    let (config,source_config,files)=source_at_shape(&sources,index,width,height);
    Ok(ResolvedPackage{config,source_config,files,schema:3,sources})
}
impl ResolvedPackage {
    pub fn select_text_tokens(&mut self,tokens:usize)->Result<(),String> {
        if tokens==0 || tokens>2048 {return Err("Prompt must contain 1..2048 tokens including BOS".into());}
        if self.schema!=3 {
            if tokens>self.config[2] as usize {return Err(format!("Prompt exceeds this model's {} token bucket",self.config[2]));}
            return Ok(());
        }
        let index=(0..self.sources.len()).filter(|&i|tokens<=self.sources[i].config[2] as usize)
            .min_by_key(|&i|self.sources[i].config[2]).ok_or("Prompt exceeds available text buckets")?;
        let (config,source_config,files)=source_at_shape(&self.sources,index,self.config[0]*16,self.config[1]*16);
        self.config=config; self.source_config=source_config; self.files=files;
        Ok(())
    }
}

pub fn open(root:&Path,width:i32,height:i32)->Result<ResolvedPackage,String>{
    if width<0||height<0||(width==0)!=(height==0)||width>2048||height>2048||width%16!=0||height%16!=0||i64::from(width)*i64::from(height)>2097152{return Err("Invalid requested WH".into());}
    let manifest=json(&root.join("manifest.json"))?;
    if manifest["schema_version"].as_u64()==Some(3){
        let contract:Value=serde_json::from_str(include_str!("../schema3_contract.json")).map_err(|e|e.to_string())?;
        let all=verify_contract(root,&contract)?;
        return select(all,&contract,width,height);
    }
    crate::package::verify(root)?;
    let cfg=config(&manifest["config"])?;
    if width!=0&&(cfg[0].checked_mul(16)!=Some(width)||cfg[1].checked_mul(16)!=Some(height)){return Err("Static package resolution mismatch".into());}
    Ok(ResolvedPackage{config:cfg,source_config:cfg,files:required_files().into_iter().map(|n|{let p=root.join(&n);(n,p)}).collect(),schema:manifest["schema_version"].as_u64().ok_or("Missing schema")? as u32,sources:Vec::new()})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn contract_keeps_reviewed_shapes_and_bn_asymmetry(){let c:Value=serde_json::from_str(include_str!("../schema3_contract.json")).unwrap();assert_eq!(c["source_manifests"].as_object().unwrap().len(),3);assert_eq!(c["reviewed_encoders"].as_object().unwrap().len(),2);assert_eq!(c["reviewed_encoders"]["ef98859ac741f6923680fb02de39e663fa3fa01943eff9d2d85c6ddaf40c9e59"]["width"],512);assert_eq!(c["reviewed_encoders"]["72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1"]["width"],1024);assert_ne!(c["math"]["encoder_bn_eps"],c["math"]["decoder_inverse_bn_eps"]);assert_eq!(c["encoder"]["status"],"unavailable");}
    #[test]fn malformed_digest_and_config_fail(){for s in ["../oops","A",&"A".repeat(64)]{assert!(sha(s).is_err());}assert!(config(&serde_json::json!({"packed_width":true})).is_err());}

    fn runtime_source() -> PackageSource {
        PackageSource {
            config:[64,64,64,64],
            source_manifest:"72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1".into(),
            files:[("dit/block-35/block.ncnn.bin".into(),PathBuf::from("objects/unchanged-weight")),
                   ("vae/encoder.ncnn.param".into(),PathBuf::from("objects/source-encoder-graph")),
                   ("vae/encoder.ncnn.bin".into(),PathBuf::from("objects/source-encoder-weight"))].into_iter().collect(),
        }
    }
    #[test]
    fn runtime_target_keeps_source_weights_and_hides_wrong_size_encoder() {
        let contract:Value=serde_json::from_str(include_str!("../schema3_contract.json")).unwrap();
        let package=select(vec![runtime_source()],&contract,1376,768).unwrap();
        assert_eq!(package.config,[86,48,64,64]);
        assert_eq!(package.source_config,[64,64,64,64]);
        assert_eq!(package.files["dit/block-35/block.ncnn.bin"],PathBuf::from("objects/unchanged-weight"));
        assert!(!package.files.contains_key("vae/encoder.ncnn.param"));
        assert!(!package.files.contains_key("vae/encoder.ncnn.bin"));
        for (width,height) in [(0,0),(1024,1024)] {
            let original=select(vec![runtime_source()],&contract,width,height).unwrap();
            assert_eq!(original.config,original.source_config);
            assert!(original.files.contains_key("vae/encoder.ncnn.param"));
        }
    }
    #[test]
    fn actual_token_count_selects_smallest_available_export_without_losing_sources() {
        let contract:Value=serde_json::from_str(include_str!("../schema3_contract.json")).unwrap();
        let sources=contract["source_manifests"].as_object().unwrap().iter().map(|(digest,c)| {
            let cfg=config(c).unwrap();
            PackageSource { config:cfg, source_manifest:digest.clone(),
                files:[("text/block-00/text.ncnn.param".into(),PathBuf::from(format!("objects/text-{}",cfg[2])))].into_iter().collect() }
        }).collect();
        let mut package=select(sources,&contract,2048,1024).unwrap();
        for (tokens,bucket) in [(1,32),(32,32),(33,64),(64,64),(65,2048),(2048,2048),(15,32)] {
            package.select_text_tokens(tokens).unwrap();
            assert_eq!(package.config,[128,64,bucket,if bucket==32 {64} else {bucket}]);
            assert_eq!(package.files["text/block-00/text.ncnn.param"],PathBuf::from(format!("objects/text-{bucket}")));
            assert_eq!(package.sources.len(),3);
        }
        let before=package.config;
        for tokens in [0,2049,usize::MAX] {assert!(package.select_text_tokens(tokens).is_err());assert_eq!(package.config,before);}
        let mut only64=select(vec![runtime_source()],&contract,1024,1024).unwrap();
        only64.select_text_tokens(15).unwrap();assert_eq!(only64.config[2],64);
        assert!(only64.select_text_tokens(65).is_err());
        let mut legacy=ResolvedPackage{config:[64,64,64,64],source_config:[64,64,64,64],files:BTreeMap::new(),schema:2,sources:Vec::new()};
        legacy.select_text_tokens(15).unwrap();assert_eq!(legacy.config[2],64);
        assert!(legacy.select_text_tokens(65).is_err());
    }
    #[test]
    fn runtime_range_keeps_source_identity_and_rejects_invalid_geometry() {
        let contract:Value=serde_json::from_str(include_str!("../schema3_contract.json")).unwrap();
        for (width,height) in [(768,1376),(1360,768),(2048,1024),(1024,2048),(16,16),(16,2048),(2048,16)] {
            assert_eq!(select(vec![runtime_source()],&contract,width,height).unwrap().config[..2],[width/16,height/16]);
        }
        for (width,height) in [(0,16),(15,16),(2048,2048),(2049,16),(1377,768)] {
            assert!(select(vec![runtime_source()],&contract,width,height).is_err());
        }
        let mut other=runtime_source();other.source_manifest="0".repeat(64);
        assert!(select(vec![other],&contract,1376,768).is_err());
        assert!(select(vec![runtime_source(),runtime_source()],&contract,1376,768).is_err());
        let mut wrong_bucket=runtime_source();wrong_bucket.config[2]=2048;
        assert!(select(vec![wrong_bucket],&contract,1376,768).is_err());
    }
}

#[cfg(test)]
mod corruption_tests {
    use super::*;
    use sha2::{Digest,Sha256};
    #[test] fn small_store_positive_and_corruption_matrix(){
        let root=std::env::temp_dir().join(format!("ernie-schema3-test-{}",std::process::id()));
        fs::create_dir(&root).unwrap();fs::create_dir(root.join("objects")).unwrap();
        let put=|bytes:&[u8]|{let digest=format!("{:x}",Sha256::digest(bytes));fs::write(root.join("objects").join(&digest),bytes).unwrap();digest};
        let data=b"synthetic weight, never a production trust entry";let weight=put(data);
        let cfg=serde_json::json!({"packed_width":32,"packed_height":24,"text_bucket":2048,"dit_text_tokens":2048,"text_layers":25,"dit_layers":36});
        let files:serde_json::Map<String,Value>=required_files().into_iter().map(|n|(n,weight.clone().into())).collect();
        let sizes:serde_json::Map<String,Value>=required_files().into_iter().map(|n|(n,(data.len() as u64).into())).collect();
        let source=serde_json::json!({"schema_version":2,"portable":true,"config":cfg,"files":files,"file_sizes":sizes});
        let source_bytes=serde_json::to_vec(&source).unwrap();let digest=put(&source_bytes);
        let mut contract:Value=serde_json::from_str(include_str!("../schema3_contract.json")).unwrap();
        contract["source_manifests"]=serde_json::json!({digest.clone():cfg});
        let mut m=serde_json::json!({"schema_version":3,"format":contract["format"],"math":contract["math"],"encoder":contract["encoder"],"generation_quality_status":contract["generation_quality_status"],"objects":{weight.clone():data.len(),digest.clone():source_bytes.len()},"instances":[{"source_manifest_sha256":digest,"config":cfg,"runtime_bindings":files}]});
        let save=|v:&Value|fs::write(root.join("manifest.json"),serde_json::to_vec(v).unwrap()).unwrap();save(&m);
        assert_eq!(verify_contract(&root,&contract).unwrap()[0].files.len(),136);
        let available=serde_json::json!({"status":"available","width":64,"height":64,"posterior":"mode","packing":"pack","encoder_bn_eps":0.0001,"encoder_bn_affine":false,"decoder_inverse_bn_eps":0.00001,"files":{
            "vae/encoder.ncnn.param":{"sha256":weight,"size":data.len()},"vae/encoder.ncnn.bin":{"sha256":weight,"size":data.len()},
            "vae/bn-mean.f32":{"sha256":weight,"size":data.len()},"vae/bn-variance.f32":{"sha256":weight,"size":data.len()}},"evidence":{}});
        contract["reviewed_encoders"]=serde_json::json!({digest.clone():available});
        let mut declared=available.clone();declared.as_object_mut().unwrap().remove("evidence");
        let files=declared["files"].as_object().unwrap().iter().map(|(n,v)|(n.clone(),v["sha256"].clone())).collect();
        declared["files"]=Value::Object(files);m["encoder"]=declared;save(&m);
        let resolved=verify_contract(&root,&contract).unwrap();assert_eq!(resolved[0].files.len(),138);
        let trusted=contract["reviewed_encoders"][&digest].clone();
        contract["reviewed_encoders"]=serde_json::json!({"0".repeat(64):trusted});
        assert!(verify_contract(&root,&contract).is_err());
        contract["reviewed_encoders"]=serde_json::json!({digest.clone():available});
        m["encoder"]["width"]=32.into();save(&m);assert!(verify_contract(&root,&contract).is_err());
        m["encoder"]=contract["encoder"].clone();save(&m);
        let original=m.clone();m["instances"][0]["runtime_bindings"].as_object_mut().unwrap().remove("dit/block-35/block.ncnn.bin");save(&m);assert!(verify_contract(&root,&contract).is_err());
        m=original.clone();m["objects"][&weight]=0.into();save(&m);assert!(verify_contract(&root,&contract).is_err());
        m=original.clone();m["instances"][0]["source_manifest_sha256"]="0".repeat(64).into();save(&m);assert!(verify_contract(&root,&contract).is_err());
        m=original.clone();m["math"]["decoder_inverse_bn_eps"]=0.0001.into();save(&m);assert!(verify_contract(&root,&contract).is_err());
        save(&original);fs::write(root.join("objects").join(&weight),vec![b'x';data.len()]).unwrap();assert!(verify_contract(&root,&contract).is_err());
        assert!(strict_json(br#"{"a":1,"a":2}"#).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
