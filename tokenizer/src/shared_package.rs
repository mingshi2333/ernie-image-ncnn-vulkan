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

pub struct ResolvedPackage { pub config: [i32; 4], pub files: BTreeMap<String, PathBuf>, pub schema: u32 }
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
pub fn verify(root:&Path)->Result<Vec<ResolvedPackage>,String>{
    let contract:Value=serde_json::from_str(include_str!("../schema3_contract.json")).map_err(|e|e.to_string())?;
    verify_contract(root,&contract)
}
fn verify_contract(root:&Path,contract:&Value)->Result<Vec<ResolvedPackage>,String>{
    let m=json(&root.join("manifest.json"))?;
    keys(&m,&["schema_version","format","instances","objects","math","encoder","generation_quality_status"])?;
    for key in ["schema_version","format","math","encoder","generation_quality_status"] {if m[key]!=contract[key]{return Err(format!("Unreviewed schema-3 {key}"));}}
    let objects=m["objects"].as_object().ok_or("Missing objects")?;
    let instances=m["instances"].as_array().ok_or("Missing instances")?;
    if instances.is_empty()||instances.len()>2{return Err("Invalid static instance count".into());}
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
        // Exact source manifest checksum binds model.cfg, all layer graphs and weights.
        // C++ additionally enforces the canonical full-graph allowlist before instantiation.
        out.push(ResolvedPackage{config:config(pinned)?,files,schema:3});
    }
    if used!=objects.keys().cloned().collect(){return Err("Unbound shared objects".into());}
    let entries=fs::read_dir(root.join("objects")).map_err(|e|e.to_string())?.map(|e| e.map_err(|e|e.to_string()).and_then(|e|e.file_name().into_string().map_err(|_|"Invalid object name".into()))).collect::<Result<BTreeSet<_>,String>>()?;
    if entries!=used{return Err("Unlisted shared objects".into());}
    let entries=fs::read_dir(root).map_err(|e|e.to_string())?.map(|e|e.map_err(|e|e.to_string()).and_then(|e|e.file_name().into_string().map_err(|_|"Invalid package name".into()))).collect::<Result<BTreeSet<_>,String>>()?;
    if entries!=["manifest.json".to_string(),"objects".to_string()].into_iter().collect(){return Err("Unlisted package entries".into());}
    Ok(out)
}
pub fn open(root:&Path,width:i32,height:i32)->Result<ResolvedPackage,String>{
    if width<0||height<0||(width==0)!=(height==0)||width>2048||height>2048||width%16!=0||height%16!=0||i64::from(width)*i64::from(height)>2097152{return Err("Invalid requested WH".into());}
    let manifest=json(&root.join("manifest.json"))?;
    if manifest["schema_version"].as_u64()==Some(3){
        let all=verify(root)?;
        if width==0&&all.len()!=1{return Err("Multiple static instances require explicit width and height".into());}
        return all.into_iter().find(|p|width==0||(p.config[0].checked_mul(16)==Some(width)&&p.config[1].checked_mul(16)==Some(height))).ok_or("Unreviewed/unavailable target shape".into());
    }
    crate::package::verify(root)?;
    let cfg=config(&manifest["config"])?;
    if width!=0&&(cfg[0].checked_mul(16)!=Some(width)||cfg[1].checked_mul(16)!=Some(height)){return Err("Static package resolution mismatch".into());}
    Ok(ResolvedPackage{config:cfg,files:required_files().into_iter().map(|n|{let p=root.join(&n);(n,p)}).collect(),schema:manifest["schema_version"].as_u64().ok_or("Missing schema")? as u32})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn contract_keeps_two_reviewed_shapes_and_bn_asymmetry(){let c:Value=serde_json::from_str(include_str!("../schema3_contract.json")).unwrap();assert_eq!(c["source_manifests"].as_object().unwrap().len(),2);assert_ne!(c["math"]["encoder_bn_eps"],c["math"]["decoder_inverse_bn_eps"]);assert_eq!(c["encoder"]["status"],"unavailable");}
    #[test]fn malformed_digest_and_config_fail(){for s in ["../oops","A",&"A".repeat(64)]{assert!(sha(s).is_err());}assert!(config(&serde_json::json!({"packed_width":true})).is_err());}
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
        let original=m.clone();m["instances"][0]["runtime_bindings"].as_object_mut().unwrap().remove("dit/block-35/block.ncnn.bin");save(&m);assert!(verify_contract(&root,&contract).is_err());
        m=original.clone();m["objects"][&weight]=0.into();save(&m);assert!(verify_contract(&root,&contract).is_err());
        m=original.clone();m["instances"][0]["source_manifest_sha256"]="0".repeat(64).into();save(&m);assert!(verify_contract(&root,&contract).is_err());
        m=original.clone();m["math"]["decoder_inverse_bn_eps"]=0.0001.into();save(&m);assert!(verify_contract(&root,&contract).is_err());
        save(&original);fs::write(root.join("objects").join(&weight),vec![b'x';data.len()]).unwrap();assert!(verify_contract(&root,&contract).is_err());
        assert!(strict_json(br#"{"a":1,"a":2}"#).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
