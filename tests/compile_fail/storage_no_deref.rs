// `Storage` has no `Deref<Target = [f32]>`: host access is spelled `to_host_f32()` or
// `as_host_f32()`, so a GPU sync can't hide behind a coercion.
use volta::Storage;

fn main() {
    let s = Storage::cpu(vec![1.0, 2.0]);
    let _: &[f32] = &s;
}
