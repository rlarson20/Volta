use volta::{Embedding, Sequential};

fn main() {
    // `Embedding` takes indices, not a tensor: it is an `IndexedLayer`, not a `Layer`.
    let _ = Sequential::new(vec![Box::new(Embedding::new(4, 2))]);
}
