use std::env;
use std::fs::{read_dir, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;

const GIT_REPOSITORY: &str = "https://github.com/airockchip/rknn-toolkit2";
const VERSION: &str = "2.2.0";
const OS: &str = "Linux";

fn main() {
    let repository_dir = download_git_repository();
    let include_dir = repository_dir.join(format!("rknpu2/runtime/{OS}/librknn_api/include/"));
    let libs_dir = repository_dir.join(format!("rknpu2/runtime/{OS}/librknn_api/aarch64/"));

    // Re-run header wrapper generation if headers changed
    let headers = read_dir(&include_dir).unwrap();
    headers.for_each(|it| {
        println!("cargo:rerun-if-changed={it:?}");
    });
    let wrapper_header_path = create_wrapper_header(include_dir);

    // Look at the right rknnrt library
    println!("cargo:rustc-link-search={libs_dir:?}");
    println!("cargo:rustc-link-lib=rknnrt");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header(wrapper_header_path.into_os_string().into_string().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn create_wrapper_header(include_dir: PathBuf) -> PathBuf {
    let wrapper_header_path = include_dir.join("rknn_api_wrapper.h");
    let f = File::create(&wrapper_header_path).expect("unable to create file");
    let mut f = BufWriter::new(f);

    writeln!(f, "#include \"rknn_api.h\"").unwrap();
    writeln!(f, "#include \"rknn_matmul_api.h\">").unwrap();

    wrapper_header_path
}

fn download_git_repository() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let repository_path = out_dir.join("rknn-toolkit2");
    println!("{out_dir:?}");
    if !repository_path.join(".git").exists() {
        println!("Creating directory {:?}", repository_path);
        std::fs::create_dir_all(&repository_path).unwrap();
        run("git", |command| {
            command
                .arg("clone")
                .arg(format!("--branch=v{VERSION}"))
                .arg("--depth=1")
                .arg(GIT_REPOSITORY)
                .arg(&repository_path)
        });
    }
    repository_path
}

fn run<F>(name: &str, mut configure: F)
where
    F: FnMut(&mut Command) -> &mut Command,
{
    let mut command = Command::new(name);
    let configured = configure(&mut command);
    println!("Executing {:?}", configured);
    if !configured.status().unwrap().success() {
        panic!("failed to execute {:?}", configured);
    }
    println!("Command {:?} finished successfully", configured);
}
