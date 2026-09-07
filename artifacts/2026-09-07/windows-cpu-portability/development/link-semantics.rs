use std::{fs, path::Path};
use std::os::windows::fs::{MetadataExt, symlink_file, symlink_dir};

fn describe(path: &Path) {
    match fs::symlink_metadata(path) {
        Ok(meta) => println!("path={:?} symlink={} file={} directory={} attributes=0x{:x} canonical={:?}",
            path, meta.is_symlink(), meta.is_file(), meta.is_dir(), meta.file_attributes(), fs::canonicalize(path)),
        Err(error) => println!("path={:?} metadata_error={:?}", path, error),
    }
}

fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() == 4 {
        let result = if args[1] == "create-directory" { symlink_dir(&args[2], &args[3]) }
                     else { symlink_file(&args[2], &args[3]) };
        result.expect("Windows symbolic link creation failed");
        let path = Path::new(&args[3]);
        describe(path);
        assert!(fs::symlink_metadata(path).unwrap().is_symlink());
        return;
    }
    let root = std::env::args_os().nth(1).expect("fixture root");
    let root = Path::new(&root);
    for name in ["target", "posix-file-link", "posix-directory-link"] { describe(&root.join(name)); }
    let link = root.join("windows-file-link");
    println!("windows_create_symbolic_link={:?}", symlink_file(root.join("target"), &link));
    describe(&link);
}
