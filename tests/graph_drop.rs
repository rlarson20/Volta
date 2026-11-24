//! Dropping a long graph must not overflow the stack (#91).

use volta::RawTensor;

/// Builds a chain of `n` adds and drops it, on a thread with a small stack.
fn build_and_drop_chain(n: usize) {
    std::thread::Builder::new()
        // About a tenth of the default test-thread stack, so a recursive drop of this
        // chain would overflow by a wide margin.
        .stack_size(256 * 1024)
        .spawn(move || {
            let x = RawTensor::new(vec![1.0], &[1], true);
            let mut acc = x.clone();
            for _ in 0..n {
                acc = acc.add(&x);
            }
            assert_eq!(acc.item(), 1.0 + n as f32, "the chain was built");
            drop(acc);
            // Every node is gone, so `x` holds the only handle again.
            assert_eq!(x.item(), 1.0);
        })
        .expect("spawn the test thread")
        .join()
        .expect("the chain dropped without panicking");
}

#[test]
fn dropping_a_million_node_chain_does_not_overflow_the_stack() {
    build_and_drop_chain(1_000_000);
}

#[test]
fn dropping_a_branching_graph_drops_shared_nodes_once() {
    // Each node is a parent three times over (twice of the add, once of the sub), so
    // the work list meets it with other handles alive, and must free it at the last.
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let x = RawTensor::new(vec![1.0], &[1], true);
            let mut acc = x.clone();
            for _ in 0..200_000 {
                acc = acc.add(&acc).sub(&acc);
            }
            assert_eq!(acc.item(), 1.0);
            drop(acc);
        })
        .expect("spawn the test thread")
        .join()
        .expect("the graph dropped without panicking");
}
