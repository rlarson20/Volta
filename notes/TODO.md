## new notes

### ML lib issues

#### blocking defects

- [ ] ConvTranspose2d weights get no grad in normal training conf
  - `conv_transpose.rs:417` gates attach to graph iff input has flag
  - in training input is data (no grad) and weights need grad
  - rn that conf has grad = None, bias trains though
  - check empirically: x.no_grad impl weight.grad = false, bias.grad = true; x.grad=true impl weight.grad = true
  - Consequence: dcgan, vae, super_resolution's generators' fc and both deconv layers have frozen weights
  - ex runs, prints loss, demonstrates nothing
  - min fix: gate on {x,weight,bias}.requires_grad, gradfn already checks per-parent req_grad
- [ ] LSTMCell fucks up autograd graph at gate slice
  - `lstm.rs:123-145`: slice gate copies gate data into RawTensor constructor w rg but no gfn or parents
  - backward stops at all 4 slices: i/h weights and biases get 0 grad, so does input
  - thf: untrainable LSTM
  - Known and enshrined: test asserts grad is None and passes with message confirming
  - worse than no test, launders defect, readme says otherwise
  - min fix: use shrink op or scatter-based SliceGradFn
- [ ] PosEncoding::forward severs graph in both modes
  - sinusoidal and learned paths rebuild output w raw data copy
  - result_data.push(x_val + enc_val) -> new tensor w last param false
  - x gets no grad, screws up embedding layers upstream
  - learned mode: pos embedding table can never receive a gradient
  - thf: char_trans_model hand-rolls pos-enc instead of using component
  - another isntance of example routing around broken internals
  - min fix: compose from graph ops (shrink table, broadcast add)
- [ ] diff-rank batched matmul backward is wrong, doesnt say shit
  - matmul:506-559 (grad wrt A) indexes `n = y.shape[y? no - x.rank()-1]`
  - uses x rank on y shape
  - mirror bug for grad wrt B at 620-668
  - feeds mismatched shapes into broadcast_to
  - silently maps OOR indices to 0.0 instead of a failure
  - fw accepts [2,3,4] @ [2,1,4,6]
  - fin-diff check gives 29% grad error (equal rank [2,3,4] @ [1,4,6] passes at 4.8e-4)
  - double check above
  - min fix: derive dim from own tensor rank
  - alt: reject unequal rank in forward
- [ ] CPU only build does not compile
  - --no-default-features fails w 5 errors
  - ops/gpu_ops.rs use crate::gpu::GpuKernels unconditionally
  - conv.rs (unguarded gpu_col2im, cfg-gated if/else arms w incomp types)
  - sgd calling step_cpu_param, invered-cfg'd out at sgd
    - GPU feature flag on CPU path: Adam did right, not SGD
  - cargo.toml says gpu is optional, claude.md tells reader to use --no-default-features for CPU-only
  - hasn't been run tbh
  - min fix: correct cfg gates, fix CI with NDF

#### High-severity

- [ ] MaxPool2d pads w 0.0 instead of -inf
  - pads w generic pad op
  - overlapping padding window returns 0.0 even when all real vals are neg
  - verify: maxpool(2,2,pad=1) over all(-5.0) input -> 0.0
  - deviates from PyTorch semantics, wrong when pooling precedes signed activations
- [ ] bce_w_logits_loss overflows, doc comment lies
  - tensor.rs:445-464 computes log(1+exp(x)) directly
  - |x| >= 88 has inf loss in f32 (check again), grads NaN
  - header states (numerically stable)
  - bce_loss has same issue w `log(p)/log(1-p)`
  - min fix: softplus `max(x,0)+log1p(exp(- |x|))`
- [ ] unsafe code has violated invariant
  - storage hands `&[f32]` to GPU `cpu_cache` with note
  - 'never clear cache once populated'
  - invalidated by `invalidate_cpu_cache()` storage:627
  - &mut borrow discipline saves this, not stated invariant
  - as_{dtype}_slice on GPU (302-385) reinterp f32 as f64/16
  - dead wrong but unreachable
- [ ] lib panics on data
  - reduce_op(Max) panics on NaN
  - xavier_transform panics on > 2D shapes (gens fan_in*fan_out elems regardless of shape prod)
  - stride op panics on stride 0 (div by 0 movement:1102)
  - ships VoltaError enum and deny-heavy clippy, panics on ordinary inputs
- [ ] GPU broadcast back silent truncates rank > 4
  - kernels:2955 pad_to_u32_4 take(4) on shapes
  - passes true rank to shaders that index `vec4<u32>` dynamically
  - OOB vector indexing in WGSL ->
  - silent wrong grads for rank-5+ tensors instead of cpu fallback
- [ ] transformerblock decorative
  - attn SimpleAttention placeholder
  - causal param ignored
  - load_state_dict skips attention state silent
  - char_transformer_model bypasses
  - doc'ed as todo admitting MultiHeadAttention makes block non-standard
  - real MultiHeadAttention exists and is correct in structure
  - embeds residual + opt LayerNorm (whole block, not layer)
  - has dead dropout field
- [ ] load_state_dict demotes GPU to CPU
  - layer impl writes storage::CPU w no device preservation, element count validation
  - load chackpoint into GPU-resident model, params move to CPU mid-training w no err
- [ ] muon impl is toy, panics on GPU params
  - uses ununed quadratic Newton-Schulz
  - frob-norm "approx spectral" scaling
  - magic 1/10 w stability comment
  - mutates via storage::iter_mut, panics for GPU backed params

#### Medium/nice to have

- [ ] every op deep copies
  - storage::clone is a `Vec<u8>` copy
  - every op in un/bin/mm/move/tern leads w `s.data.clone()`
  - broadcasts are materialized (broadcast_to, no stride-0 view)
  - elementwise op allocates ~3x output size
  - WhereGradFn keeps full broadcast-sized cond for graph lifetime
- [ ] im2col copies whole tensor per kernel elem
  - let cpu_data = data.to_vec, inside innermost kw_idx loop
  - O(BHWCK^2) full tensor copies
- [ ] conv input grad is quadratic
  - should be linear in K^2
  - conv.rs:427-518,822-912 scan all H_out*W_out out pos per in pixel (receptive-field test)
  - should iterate the <=K^2 contrib positions
  - at 64x64, K=3, ~450x wasted inner loop work
  - tiled iGEMM (1679-1782) has no tile reuse
  - direct conv w extra bounds check
  - 'good balance of speed and memory' is label, not property
- [ ] gpu reductions are single threaded
  - reduce.wgsl runs seq loop at workgroup_size(1)
  - gpu_max_reduce downloads to CPU to find argmax (gpu_ops.rs:131)
  - every GPU op builds fresh uniform buffer, bind group, encoder, submit
- [ ] 2-pass 'race free' broadcast backward's pass 2 is O(targets * out_size)
  - brute force scan per target element
- [ ] check_grads doesn't 0 pre-existing grads
  - tensor.rs:898-902
  - reusing a tensor or calling after backward produces false failures
  - check (said verified w err 1.02 on proper gradient)
- [ ] committed flakey test on conv.rs
  - systematically loosened tolerances (ep=5e-2, tol in \[2,5\]e-2) on conv FD checks
  - GPU forward-consistency tests use all-ones ins, weights
  - masks most indexing bugs
  - exception: col2im tests use varied data
- [ ] Sequential::load_state_dict prefix match is greedy
  - key "10.weight" matches "1." in seq.rs starts_with loop
  - leaks bogus 0.weight entry into layer 1 substate
- [ ] RNG repro holes
  - dataloader shuffles with rand instead of seeded global RNG
  - with_rng silently falls back to unseeded thread RNG under reentrant borrow
- [ ] io/config is dead file of non-compiling pseudocode
  - bodiless fn decl
  - derive deser w no serde dep
  - escapes compiler only bc no module declares
- [ ] dtype system is decorative
  - dtype supports 8 types
  - every compute path is f32
  - to_device downcasts to f32
- [ ] embedding::forward takes `&[usize]`
  - impl module for embedding panics by design
  - a seq holding one detonates at runtime
- [ ] mixed device bin ops silent make cpu out
- [ ] f64 on gpu reinterprets bytes (see H3)

##### WGPU backend

real, not decorative
genuine context w ~70 pipelines
pooled buffers, staged readbacks
properly tiled 16x16 shared-mem matmul
working conv fw/bw kernels w CPU parity tests
GPU optimizer step whose adam matches (valid) CPU one

coverage against CPU path sucks
reductions single threaded theater
broadcast forward falls back to CPU
broadcast backward caps at rank 4 (H5)
arrayLength-sized shaders overrun into 2^n bucket padding
harmless only bc of unenforced invariant holding by accident:
buffers always rewritten [0..len) before logical reads

##### tests

this is not no tests
641 passing tests w finite-diff grad checks across ops,
cross-algo equiv, GPU/CPU parity w varied data
issue is placement, not volume:
0 for multi-consumer accumulation (flagged soft spot, deserves regression test)
0 tests diff w.r.t weights thru ConvTranspose (only input)
test asserting LSTM bug as expected
0 for PosEnc grads
0 for unequal rank matmul
examples bypassing broken components so tests stay green

#### finding I'd lead with

engine is correct, layers aren't, and silently
test suite is structured so it can't tell difference
diamond/accumulation code and adam flagged soft spots couldn't be broken (ill try harder)
ConvTranspose2d trains only biases in GAN/VAE examples
LSTMCell and PosEnc are grad dead ends
tests stay green because diff wrt inputs (working path) or assert bug as expected
current status would end the interview with true answers

test disciple that fixes whole class of issues:
for every layer
run loss.backward w input as plain data

```rust
assert!(weight.grad.is_some_and(|g| g.iter().any(|v| v != 0.0)))
```

single test, added to all 19 layers
catches B1-3 before anyone else had to

### Rust issues

- [ ] modern-edition codebase wrapped behind pre-1.0 style interior mut design
  - no invariant a type system would protect
  - 3 competing error strats, winner being silence
  - allow-riddled lint regime documenting hopes not practice
  - CI w step that doesn't run config it's named after
  - defensible as learning artifact ATM
  - in interview, first borrow_mut in example ends any 'well-encapsulated' convo

#### type model

- [ ] Rc RefCell around struct of data, shape, grad, req_grad, grad_fn, parents, dev
  - all relied on invariants are convention
  - can perform surgery on a tensor if you have a ref
  - own library does it constantly
    - `x.borrow_mut().requires_grad = true;` documented way to make tensor trainable
      - conv tests, ex, probe done to write these comments
    - im2col swaps storage without touching device
      - pattern appears in 10 tests
    - load_state_dict writes t.data and t.shape.clone_from as independent assigns, no elem count check
      - linear, conv, batchnorm
  - not a style quibble
  - B1 and H7 are direct R/W of pub fields from wrong place
  - opaque tensor makes bug unexpressible by accident
- [ ] accidental refcell discipline
  - each op does borrow-clone-drop dance before compute
  - a.add(&a) and x.elem_mul(&x) work today
  - BinaryGradFn::backward holds x/y_ref.borrow across entire grad comp
  - no structural prevention of re-entrant op (legit pattern in VJP, call fw inside bw) from panicking w BorrowMutError
  - design safety requires remembering dance
- [ ] 3 incompatible flavors of interior mut
  - refcell for tensor state
  - cell for conv2d so it can mutate config thru shared ref
  - refcell opt vec u8 inside gpu storage
  - set algo mean 'immutable' module fw pass can change behavior btwn calls
  - signals same philosophy: immutability is whatever curr borrow allows
- [ ] rawtensor:
  - clones grad state
  - rc clones parents vector
  - clone boxes grad function
  - duplicates graph topology
  - cheap write, expensive misunderstand
- [ ] not send/sync but has mutexes
  - buffer pool wraps buckets in mutex
  - mutex protecting state only 1 thread can reach is future proofing part that doesnt need it
  - actual blocker Rc is untouched
- [ ] repro is best effort
  - with_rng tries try_borrow_mut on thread-local RNG
  - if already borrowed, falls back to unseeded SILENTLY
  - manual seed guarantees nothing under reentrancy
  - DataLoader sidesteps global RNG entirely
  - seeded runs not repro over epochs

#### trait design/public API

- [ ] TensorOps 140 lines of boilerplate
  - 40 methods decl on trait
  - impl for type alias by forwarding each to RawTensor static
  - inherent methods on RawTensor plus deref free conv wrapper would have done same thing
  - now every example uses use volta::TensorOps
- [ ] 3 ways to call shit
  - static
  - trait
  - free funcs re-exported from crate root
  - pick a lane
- [ ] module trait broken by own implementors
  - impl Module for Embedding panics by design
  - seq stores boxed dyn mod and calls trait method (dyn dispatch)
  - embedding in seq is runtime bomb compiling fine
  - module for LSTMCell returns only h, silent discards cell state
  - trait forced lossy sig, impl lied to fit
  - linear has inherent fw, and trait impl whose body is self fw
  - resolves to inherent method by precence rules
  - works, reads like recursion, but confuses first maintainer changing 1 sig
- [ ] grad machinery is public
  - lib exports gradfn traits
  - internals no user can construct meaningfully
  - conv2d weight is private, linear/embedding weight are public
  - no coherent rule for what API is
- [ ] API tells that accumulate:
  - check grad returns (f32, f32, bool)
  - bool is pass fail
  - floats are max/mean err
  - nothing stops caller from ignoring failure
  - probe demonstrated related flaw of not 0ing stale grads
  - use RawTensor as new_tensor: constructor shaped alias for struct itself
  - private vocab (this needs more analysis)
  - try_/panic twin pattern exists for 4 funcs: new, sum_dim, shrink, he_init
    - mm, bc_shape, reshape, perm, ops likely to see user-shaped data, panic only

#### error strategy

- [ ] 3 regimes co-exist
  - result with well-formed voltaerror, used by the try_ cons and nothing else
  - panics: asserts for every shape, broadcast, dim, channel check in op layer
  - 458 non-test unwrap/expect occurrences in src/
  - biggest one: silence
- [ ] lint config made correctness worse
  - indexing_slicing = deny pushes kernel code to 77 sites with unwrap_or style access in math kernels
    - need to investigate further ngl
  - bounds violation (usually index map bug) produces silet 0 instead of panic
  - not hypothetical: B4 stayed silent bc broadcast_to maps OOR source indices through data.get(from_idx) => default (check bin.rs:519-523)
  - loud-panic policy causes bug to fail on 1st run
  - curr policy: ships green behind 641 test suite
  - lint turning corruption into wrong numbers is shit for numerical code
- [ ] GPU layer returns option for everything
  - callers convert none into warnings plus silent cpu fallback
  - eprintln req but failed in mm, bin, tern, conv
  - users op silently changes dev, kernel class, threading chars
  - only signal is stderr may be buried in logs
  - mixed dev bin ops dont warn: devices are unwrapped and discarded and computed on CPU (mm:137-154)
- [ ] library prints
  - 47 print/eprintln sites in non-test source
  - per-call fallback warnings that will spam in training loop
  - printing to stdout unprompted is a defect for a crates.io library
- [ ] one good thing: missing docs are denied and honored
  - documentation discipline is real, policy being documented is wrong

#### unsafe audit

- [ ] volume is low
  - hand decl extern "C" cblas_sgemm w manual i32 casts
    - works, but blas-src is a dep, supplies decl
    - hand rolled extern next to bindings crate is drift waiting to happen
  - matrixmultiply::sgemm unsafe reqd by API, args, strides correct
  - bytemuck casts through storage.rs, tool used right
- [ ] storage block
  - safety comment on try_as_f32_slice falsified by invalidate_cpu_cache
  - soundness rescured only by borrow checker discipline
  - invalidation requires mut ref, cant coexist w live slice
  - accident of API shape, not documented reasoning
  - false safety is worse than none bc refactor trusts it
  - same file keeps as dtype slice fns for gpu storage that reinterpret f32 bytes
  - unreachable but wrong copies of cache pattern

#### ownership and clone economy

- [ ] own notes count 400+ clone ops across 45 files
- [ ] load bearing pattern repeated by hand every op:
  - deep copy
  - compute into fresh vec
  - borrow mut clone t into output parents and define grad fn in a box
- [ ] 3 issues at design level
  - cpu storage clone is deep VecU8 copy, clone economy isn't Rc-pointer economy at all on CPU path
  - borrow dance never abstracted
    - no with_tensor helper, no op builder
    - same 5 lines appear dozens of times, requiring clippy deny table
    - deref impl for storage is deref polymorphism over diff types
      - data.iter can panic, block and iter/deref_mut panic on gpu storage
      - exact mechanism behind muon panicking (H8)
    - explicit try_as_host_f32 would put cost and failure at call site instead of 3 layers below it

#### abstraction quality

- [ ] Copy-paste structs
  - BatchNorm1d/2d are duplicated, differing in reduction axist
  - generic over axis impl is textbook fix
  - already forced || edits (same bessel-correction logic had to be typed 2 times)
- [ ] lint culture is performative
  - 15 allow too many lines
  - 9 expects with reasons being TODO confessions
  - crate level allow float_cmp, similar_names, cast_precision_loss
  - 60 line deny table plus scoped exceptions like this ==> aspiration, not code
- [ ] 16 pos usize args on GPU kernel wrappers
  - WGSL side uses Im2colParams and MovementParams structs
  - had good idea in shaders, didnt use in rust
- [ ] cfg triplication
  - DirectConvGradFn dupes each cpu call across cfg arms
  - structure that produced 2/5 no features compile errors (B5)
  - correct config hygiene would compute arms once
- [ ] config.rs
  - dead file of pseudocode
  - survives bc it doesn't get pulled in
- [ ] lib.rs: 83 lines of exp, ~5000 lines of test, alongside 17 integration test files
  - follows no rule (comment of lib:4094 admits convention)
  - blob makes crate root largest file in project
- [ ] diagnostic tooling exported as API
  - pub on early_warning, system_monitor, monitor
  - system_monitor shelling to ps is test scaffolding not library surface

#### deps and packaging

- [ ] default has heavy wgpu by default
  - downstream crate wanting cpu autograd pays full wgpu compile unless it opts out
  - readme frames GPU as opt-in, contradicting
  - claude admits mismatch: docs know, manifest doesn't
- [ ] statrs pulled in for erf
  - 20 line Abramowitz-Stegun or libm would delete several dependencies
- [ ] approx in dependencies but only used in tests, so it's a dev dependency
- [ ] blas-src dep only to link Accelerate
  - actual cblas decl handwritten
  - linkage and decl can drift independently
- [ ] it is modern: 2024, 1.89 vers, let-chains, crate metadata, best maintained dim of project
- [ ] repo hygiene:
  - no tracking target, but pub repo carries LLM-session artifacts at top level
  - responses/*.md
  - sys prompts
  - unfinished minigrad thing
  - not shipped in crate, but is repo pub face
  - node_modules at crate root invites accidents

#### CI

- [x] no-features doesn't check no-features
  - -D warnings caused allow with excuses culture instead of fixing shit
  - warnings were errors, every issue gets a waiver rather tahn change
  - GPU feature check commented out (runners tend to lack GPUs)
  - cfg gated code paths had no compiler looking from any direction
  - DONE: ci.yml split into `cpu-only` (ubuntu, real `--no-default-features` on
    clippy/test/doc + `cargo hack check --feature-powerset --exclude-features gpu`)
    and `macos` (apple silicon, `--all-features` incl gpu + full cargo-hack powerset).
    `cpu-only` job stays RED until B5 (CPU-only build does not compile) is fixed —
    that's the 4 cfg one-liners in the blocking-defects list above, not done here.

#### Minimum set for idiomatic

- [ ] Tensor needs to be opaque
  - newtype over RcRefCell w private fields, methods for every mutation
  - some mechanical work deletes invariant by convention bug class (b1, h7 can't be expressed)
- [ ] Pick error regime
  - result makes sense for shape, dim, broadcast failures
  - enum exists already
  - replace unwrap_or(0.0) sites in kernels w loud failing bounds logics
  - keep panics for interal logic errs only
- [ ] quit printing from lib
  - gate gpu init and fallback messages behind rust_log/feature flag
  - gpu dispatch failures results or doc fallback contract rather than stderr + silent cpu
- [ ] extract op-construction dance into helper
  - borrow, snapshow, compute, attach
  - adopt params struct pattern on rust side of wrappers
- [ ] fix CI's named config
- [ ] move approx to dev dep
- [ ] replace statrs for erf
- [ ] remove io/config.rs
- [ ] rewrite storage.rs unsafe block as owned data or a OnceCell-based cache w real invariant
- [ ] delete f64/16/bf16 on GPU accessors

One finding to lead with:
indexing_slicing deny is whole story in miniature
def programming lint table, adopted from blog post
converted loud bounds panics into 77 silent unwrap_or(0.0) in num kernels
that mechanism is why the worst bug shipped green under 641 passing test
culture thesis: rigor performed where lints can see it
actual failure modes live a layer below
what can i break with a borrow_mut, here, it's everything

### Preventative arch for Volta

The one-sentence summary of this whole document: **Volta's bugs were all writes that shouldn't have been possible, made silently** — so the fix is not more care, it's removing the write, typing the value, and making the silence loud.

#### Tracing the issues:

Defect: B1 ConvTranspose grad gate
Class: graph attach gated on wrong input
Prevented with: single graph edge constructor

Defect: B2/3 LSTM/PosEnc severed graph
Class: layers building raw RawTensor::new copies
Prevented with: no_grad plus private constructors

Defect: B4 unequal rank matmul grad
Class: dims derived from wrong tensor rank
Prevented with: shape/axis newtypes

Defect: B5 no-features build broken
Class: cfg triplication
Prevented with: single source cfg dispatch

Defect: H1 MaxPool 0-padding
Class: semantic constants are implicit
Prevented with: PadValue

Defect: H3 unsafe cache invariant
Class: hand-rolled unsafe on false invariant
Prevented with: OnceCell storage

Defect: TransformerBlock decorative
Class: trait cant express attention
Prevented with: trait split

Defect: Silent zeros
Class: bounds failures become data
Prevented with: loud indexing policy

Defect: silent gpu-cpu fallback
Class: option -> warning -> behavior change
Prevented with: DevicePolicy

Defect: Clone economy
Class: deep-copy-per-op ownership design
Prevented with: arena and arc storage

Defect: gradfn boilerplate / clone_box
Class: 30 lines per op, vtable/dynamic dispatch
Prevented with: closed enum + define_ops!

Defect: Tests assert the bug (LSTM)
Class: tests can encode the defect
Prevented with: assert_trainable! + cargo-mutants

Defect: embedding panics in sequential
Class: trait contract violated at runtime
Prevented with: sealed, split traits

Defect: CI step that doesn't run its name
Class: config drift
Prevented with: feature matrix

#### making illegal states unrepresentable

##### opaque tensor newtype

Prevents: B1, H7, entire invariants are conventions class
rn: Tensor with public fields means the borrow_mut.requires_grad = true is documented idiom,
load_state_dict is uncheckable field surgery

How to fix:
make tensor a wrapper around rawtensor, keeping the fields private
then implement specific ways to mutate:

```rust
pub struct Tensor {
    inner: Rc<RefCell<RawTensor>>,  // fields now private
}

impl Tensor {
    // The only ways to mutate. Each is named, findable, greppable,
    // and can carry debug_assert! invariant checks in ONE place.
    pub fn requires_grad(&self) -> bool { self.inner.borrow().requires_grad }
    pub(crate) fn set_requires_grad(&self, on: bool) { /* + invariant asserts */ }
    pub fn detach(&self) -> Tensor { /* new leaf sharing data */ }
    pub(crate) fn accumulate_grad(&self, g: Storage) { /* the ONLY grad writer */ }
    pub(crate) fn snapshot(&self) -> TensorSnapshot<'static> { /* owned clone of fields */ }
}
```

load_state_dict becomes later.load_weight(key, &data)
method validating number of elements
preserves device
instead of every layer hand-writing t.data = Storage::cpu() (rn 12 copies)
conv transpose bug stops being expressible
layer never touches requires_grad, hands parents to constructor and flag computed

cost: mechanical churn across ops (borrow.shape.clone becomes shape), roughly a weekend
behavior identical fi done before all else
prereq for nearly everything else
do first or don't bother
rule of thumb going forward:
every time you write borrow mut outside autograd and src/ops
found a missing method on tensor

##### shape and axis newtypes

prevents: B4, every dim-mixup, vec usize ambiguity making batched matmul bug silent

shapaes are vec usize today
passed loose
re-derived from wrong tensor
(matmul.rs:506-559 reads n from y's shape at x's rank)
wrap where misuse becomes type error or single checked constructor error

```rust
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// The ONLY constructor that reaches user code. Rejects zeros,
    /// enforces the 100M cap, and is the one place to debug_assert
    /// the data.len() == numel() invariant at attach time.
    pub fn from_dims(dims: impl Into<Vec<usize>>) -> Result<Self, VoltaError>;
    pub fn rank(&self) -> usize;
    pub fn numel(&self) -> usize;
    pub fn strides(&self) -> Strides;               // another newtype
    pub fn broadcast(&self, other: &Self) -> Result<Self, VoltaError>;  // the try_ twin, finally
    pub fn dim(&self, axis: Axis) -> Result<usize, VoltaError>;         // rank-checked access
    pub fn truncate_rank(&self) -> &[usize];        // explicit, named demotion
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Axis(pub usize);   // `sum_dim(Axis(1))` — an Axis can't be confused with a size
```

B4 dies structurally:
matmulSpec::new(a: &Shape, b &Shape) -> Result MatmulSpec,VoltaError
computes and validates `m,k,n,batch` once
bw reads spec, not y.shape.get(x.rank-1)
rank mismatched product is err and forward time,
not 29% wrong grad at bw time
cost: large but mechanical sig churn
worth: vec shape bugs are most productive bug family

##### repr transparent index newtypes everywhere indices meet

TensorIndex(u32)
NodeIndex(u32)
Channel(usize) where disambiguates
0 runtime cost
repr documents that
api taking 2 same primitive params meaning diff things
(batch sie, channels, 14 per gpu wrapper)
api is asking for transposition bugs
newtype cheaper than the errors

##### one graph-edge constructor

prevents: B1-3, every future op attaching grad fn under wrong cond
every op rn handwrite attach ritual
each occurrence is a chance to gate on wrong flag,
forget parents, or set flag w/o fn
25 copies of it, replace with one:

```rust
// src/autograd.rs — the only code allowed to write grad_fn/parents.
pub(crate) fn edge(
    inputs: impl AsRef<[Tensor]>,
    data: Storage,
    shape: Shape,
    grad: GradFn,          // §4.1 enum
) -> Tensor {
    let requires_grad = inputs.as_ref().iter().any(|t| t.requires_grad())
        && !grad_is_trivial(&grad);        // Mod/Cmplt return false here, once
    // attach parents + grad_fn iff requires_grad, in this function only
}
```

Consequences:
b1 dead: convtrans2d fw calls edge
no if x.requires_grad to write incorrect
any input requires grad only semantics existing

b2/3 dead:
both needed rawtensor new because graph path harder than raw path
with rawtensor new demoted and edge being easy as raw
easy and correct become same way
slicing a gate becomes gates.shrink because that's now shortest code that compiles
borrow dance boilerplate collapses into edge too

pattern that pytorch's node/edge construction centralizes
its absence caused 3 layers to ship grad-dead

##### RAII no_grad guard

prevents: check_grads stale grad contamination, batchnorm running-stat pollution during grad check, future "inf path accidentally builds graph" class

```rust
static NO_GRAD: Cell<bool> = Cell::new(false);

pub fn no_grad() -> NoGradGuard {
    NO_GRAD.set(true);
    NoGradGuard
}
pub struct NoGradGuard;
impl Drop for NoGradGuard { fn drop(&mut self) { NO_GRAD.set(false) } }

// edge() consults NO_GRAD; ops don't.
```

batchnorm's running stat update becomes let _g = no_grad(); update_stats
mut is scoped and vis
check grad opens w zero grad and no grad around num eval so loss_fn side eff cant poison comp
also natural place to add torch.inference_mode style behavior later

#### Typestate and phantom types

reflexive answer to prevent wrong-state bugs
take 2 cheap ends, decline expensive middle

##### Decline: full device typestate

full phantom-typed devices infect signatures
force either duplication or generics-with-trait-bounds on all layers
fight design where tensors more device mid-graph
other libs keep device runtime-dynamic for this reason

take instead a typestate policy:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevicePolicy {
    /// Mixed devices or missing kernels are Err(DeviceMismatch). Default.
    Strict,
    /// Fall back to CPU on kernel miss; returns the policy decision.
    PreferGpu,
    CpuOnly,
}
```

kills silent fallback class
bin-op takes policy (thread-local w set_device_policy for ergo)
PreferGpu fallbacks return FallbackEvent training loop can count and assert on
strict (default) converts H5-class silent wrong scenarios into errs
same runtime cost, all the prevention, no generic signature crap

##### Decline: Variable/Tensor typestate for requires_grad

apparently was a pytorch thing early
merged away bc dynamic control flow makes distinction leaky
every if on data vals changes type you wish you had
comp time guarantee illusory moment call graphs are dynamic
keep requires_grad runtime flag
take tiny bounded version:
bw only callable on Scalar newtype (?)

```rust
impl Tensor {
    pub fn into_loss(self) -> Result<ScalarLoss, VoltaError>;  // Err if numel != 1
}
impl ScalarLoss {
    pub fn backward(&self);   // the assert!("doesn't require grad") panic becomes a Result
}
```

converts autograd:43-46 panic into type-checked flow at 0 ergo cost
loss.into_loss()?.backward() reads fine (?)

##### Take: ZST markers where you have implicit state

ZSTs give named, checked state for free
exists in activation layers, ops that are fieldless enums
extend the discipline

```rust
pub struct Cpu;
pub struct Gpu;                 // phantom markers for a future Backend trait (§3.5)
pub type CpuTensor = Tensor;    // document intent at alias sites, no runtime change

pub struct PadValue(f32);        // §2.4
impl PadValue {
    pub const ZERO: Self = Self(0.0);
    pub const NEG_INF: Self = Self(f32::NEG_INFINITY);
}
```

PhantomData(D) on any future backend-generic struct costs 0, docs param
rule: bare primitive crossing API boundary carrying semantics (0.0 meaning zero-pad vs neutral pad),
wants to be ZST constructible newtype

##### Take: PadValue

Prevents: H1
MaxPool2d pads w 0.0 because it reused generic pad op who had pad value 0 implicit in impl
make param of op:

```rust
impl Tensor {
    pub fn pad_with(&self, pads: &[(usize, usize)], value: PadValue) -> Result<Tensor, VoltaError>;
}
// MaxPool2d::pool_forward calls pad_with(..., PadValue::NEG_INF)
```

H1 becomes unrepresentable
noone can call pooling path w/o choosing pad value
correct one is named constant at call site
not impl detail 2 layers down
same treatment for future ops with hidden scalar semantics
(eg leaky relu slope, elu alpha: put in op spec from day 1)

##### Take: Closed GradFn enum instead of Box<dyn GradFn>

Prevents: clone_box boilerplate tax (all gradfn has it), vtable dispatch in hot backward, unhandled new ops!!!

```rust
#[derive(Clone)]
pub(crate) enum GradFn {
    Unary(UnaryOp),
    Binary(BinaryOp),
    Matmul,
    SumDim { shape: Shape, axis: Axis, keepdim: bool },
    Conv { spec: ConvSpec },       // §6: the struct that replaces 16 positional usizes
    Im2col { spec: ConvSpec },
    Movement(MovementOp),
    Custom(CustomGrad),            // escape hatch: keeps the trait for exotic cases
}
```

bw becomes exhaustive match
deny wildcard_enum_match_arm in cargo ==>
compiler force contributor to handle existing var when touching this match
force adding variant's arm when adding an op
the lint becomes loadbearing
clone can be derived
clone_box disappears
enum fits smallvec style inline storage later if profiling ever cares

tradeoff: downstream crates must use gradfn custom
same escape hatch torch has (custom autograd funcs)
right default for framework

#### Trait design

##### Seal every trait meant for internal extension

Prevents: downstream half-impls of Module/TensorOps/GradFn breaking invariants assumed by lib
(eg, params and state_dict agree on keys)
convention today, enforced by no-one

```rust
pub mod sealed { pub trait Sealed {} }   // deliberately undocumented

pub trait Module: sealed::Sealed {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    // ...
}
impl sealed::Sealed for Linear {}
impl sealed::Sealed for Conv2d {}
// ...
```

buys right to add provided methods,
change signatures within 0.x w/o semver-breaking well-meaning downstream impls
pair with trybuild test proving impl module for OutsiderType fails to compile
verify a seal, don't believe it

##### Split module by input contract (prevents H6 and embedding panic)

currently forces 3 unrelated contracts through 1 method
why embedding's impl panics by design
why LSTMCell drops cell state silently

```rust
pub trait Layer: sealed::Sealed {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn state_dict(&self) -> StateDict;
    fn load_state_dict(&mut self, &StateDict) -> Result<(), VoltaError>;
    fn train(&mut self, mode: bool) { }
}

/// Embedding lives here, NOT behind Box<dyn Layer>.
pub trait IndexedLayer: sealed::Sealed {
    fn forward_indices(&self, idx: &[u32]) -> Tensor;
}

/// LSTM lives here; the type system now carries (h, c) instead of discarding c.
pub trait RecurrentLayer: sealed::Sealed {
    type State;
    fn initial_state(&self, batch: usize) -> Self::State;
    fn step(&self, x: &Tensor, state: &mut Self::State) -> Tensor;
}
```

Sequential stores Box dyn Layer and now type error to put embedding in one
alt worth considering if dtypes become more than decorative:
keep it a layer by accepting I64 dtype index tensor
that's pytorch shape of API

gives trans block place to be honest (H6)
either holds Box dyn Layer attn that must be real
no SimpleAttention id shim compiling
or attention is generic, and placeholder can't grow

##### non_exhaustive on everything public and growable

VoltaError, DType, ConvAlgo, Device, config struct
free RN, expensive to retrofit later
voltaerror esp: will want new vars
(DeviceMismatch, KernelUnavailable, RankMismatch)
and every addition == breaking change downstream unless non_exhaustive NOW

##### Delete TensorOps

With opaque tensor of 1.1, trait's reason to exist (methods on type alias) evaporates
make them inherent
if you want named import surface for discoverability, keep sealed, re-export only extension trait
140 line blog 1009-1208 in tensor should not survive

##### Later: Sealed Backend trait (sketch only)

Endgame for CPU/GPU split is burn's shape
sealed trait w associated tensor type
Cpu/Gpu ZST implementors from 2.3
curr storage becoming Cpu backend's internals
Don't start here
start with DevicePolicy
let seams form
every new GPU wrapper should be written as if it exists
(take context/params struct, return Result)
eventual migration should be move, not rewrite

#### Common trail impls

Impl: Debug (custom)
For: Tensor
Why/Guardrail: Half exists already, keep shape/dtype/grad-flag, not full data (100M param debug print == DoS on own logs). Debug on Shape, Axis, specs: derive.

Impl: Display
For: Shape, Device, DType
Why/Guardrail: "2x3"/"cpu" for human read err messages

Impl: `TryFrom<(Vec<f32>, Shape-ish)>`
For: Tensor
Why/Guardrail: Constructor story; try_new exists, TryFrom is what people reach for, lets ? work at call sites

Impl: `FromIterator<f32>`
For: Tensor builder
Why/Guardrail: only w explicit shape step. bare collect can't know shape, so wants to be TensorBuilder instead

Impl: `Index<[usize; N]>` / `Index<&[usize]>`
For: Tensor
Why/Guardrail: Checked co-ord indexing returning f32 (copy). kills tempt to hand-roll linearization at use site, #1 source of index math bugs in layer code

Impl: DO NOT IMPL PartialEq
For: Tensor
Why/Guardrail: Float eq on tensors is footgun factor. provide allclose(&self, other, rtol, atol) -> result () mismatch plus assert close test macro. Yes impl on shape/axis/specs

Impl: Default
For: TileConfig, BufferPoolConfig, optimizer hyper-params
Why/Guardrail: TileConfig default already exist. Derivable default configs should derive, and bool-heavy cons want the builder

Impl: Error
For: VoltaError
Why/Guardrail: already via thiserror, add non_exhaustive and source chain for GPU error types, curr display w/o impl std::error::Error (gpu/context 40-57)

Impl: `#[track_caller]`
For: every panic-adjacent helper
Why/Guardrail: when bounds assert fires in kernel, callers line only useful one, cheap but makes debug great

Impl: Ser/Deser (feature gate)
For: Shape, StateDict
Why/Guardrail: behind serde feature, keeps core comp lean, makes statedict interop w ecosystem (may be a way to replace bincode)

Impl: ReprTransparent and `From<inner>`
For: all 1.3 newtypes
Why/Guardrail: interop w usize boundaries w/o ceremony

Bool-param constructors: eg `Conv2d::new(in, out, k, s, p, use_bias)` six positional params where 2 are indistinguishable scalars and one a bool:
replace with builders:

```rust
Conv2d::builder(in_ch, out_ch, 3)     // required
    .stride(2).padding(1).bias()      // optional, named
    .build()?
```

precedent: SequentialBuilder
codebase knows the pattern
just doesnt apply where param lists are longests
clippy config denies fn_params_excessive_bools
builder === how you comply without allow

#### Macros

##### `define_ops!` one table per op family, emitting fw, bw, and kernel name

Prevents: fw/bw drift, CPU/GPU kernel name drift, boilerplate keeping ops written wrong (root of B2/3)
rn: unary lives in 4 places: UnaryOp enum, forward match, backward match, 2 separate kernel name tables.
5 places to forget each other
1 table:

```rust
define_ops! {
    // variant, forward expr, backward expr (g = out_grad, x = input), gpu kernels
    Neg   => |x| -x,                     |g, _x| -g,                        ("neg", "neg_backward");
    Exp   => |x| x.exp(),                |g, x|  g * x.exp(),              ("exp", "exp_backward");
    Sqrt  => |x| x.sqrt(),               |g, x|  g / (2.0 * x.sqrt()),     ("sqrt", "sqrt_backward");
    ReLU  => |x| x.max(0.0),             |g, x|  if x > 0.0 { g } else { 0.0 }, ("relu", "relu_backward");
    // ... 9 more rows. The whole unary family: 13 lines.
}
```

expands to enum, forward arms, backward arms
critically: 1 func mapping op to gpu kernel pair
unary kernel name(op) and shader pipeline table are gen from same row
SiLU becomes one line, WGSL entry
forgetting bw arm or kernel is compile error, not slow CPU fallback

shape works for bin ops (bin op family, kernel tables)
move ops keep per op structs (have state), should share gen dispatch match

cost: declarative macro_rules only

##### `param_test!` collapse test matrices

conv.rs test mod is ~2k lines of near id param test
repetitive ones become:

```rust
param_test! {
    conv_forward_shape: Conv2d forward shape in=(in_ch, out_ch, k, s, p, h, w) expect=(oh, ow) => [
        (3, 16, 3, 1, 1, 32, 32, 32, 32),
        (3,  8, 3, 2, 1, 16, 32,  8, 16),
        (3, 64, 5, 2, 2, 32, 32, 16, 16),
    ]
}
```

same asserts, one fail message format w full param tuple, adding conf to matrix is row edit
matters beyond tidiness, ConvTranspose tests never covered weights-grad path bc writing 30 line test body each time sucks.
cheap == exhaustive with tests

##### `assert_trainable!` - macro to catch B1-3

one macro, applied to each layer, encodes prop that caused 3 blockers by missing

```rust
#[macro_export]
macro_rules! assert_trainable {
    ($layer:expr, $input:expr) => {{
        let output = $layer.forward(&$input);
        let loss = output.sum().into_loss().expect("layer output must be differentiable");
        loss.backward();
        for (i, p) in $layer.parameters().iter().enumerate() {
            let g = p.grad().unwrap_or_else(|| panic!(
                "parameter {} has no gradient — graph severed or gate bug", i));
            assert!(
                g.iter().any(|v| v.abs() > 0.0),
                "parameter {} gradient is identically zero", i
            );
        }
    }};
}

// In every layer's test module:
assert_trainable!(ConvTranspose2d::new(2, 4, 3, 1, 1, true), rand_input(&[1, 2, 6, 6]));
assert_trainable!(LayerNorm::new(vec![64]),               rand_input(&[2, 8, 64]));
// Embedding via its IndexedLayer test path, LSTM via a 2-step unroll.
```

note the encoding: input ! requires_grad
real training conf, one every existing ConvTranspose test forgot
0 grads under a sum loss means severed graph or wrong gate
green run of this macro across all 19 layers is regression fence across B1-3 class

##### small test ergo

tensor macro,
assert_close (moves approx to dev dep)
gradcheck wrapper zeroing grads first
fixes check grad contamination footgun at each call site

##### Skip (for now): proc-macros

deriving GradFn from struct definitions, TVM-style op DSL,
or derive based module registration would each add comp time dep
debug tax that isn't justified at this scale
macro rules covers a lot at 0 dep cost
revist only if op count triples

#### Arena based graph - structural endgame

Prevents entirely: ref-cycle risk (avoided w luck), clone econ, Rc blocking Send, double-borrow panics under re-entrant bw

RcRefCell per node design makes every tensor an independently locked shared mutable node
textbook alt for dyn graphs on single thread:

```rust
pub struct Graph {
    nodes: Vec<Node>,               // Node { data, shape, grad, parents: SmallVec<[NodeId; 3]>, grad_fn: GradFn }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct NodeId(u32);             // index, not pointer

pub struct Tensor {
    graph: Rc<Graph>,               // one Rc per tensor, not per node
    id: NodeId,
}
```

buys:
can't build cycles
parents are indices into a vec owned by 1 graph
cant make ref cycle in index
curr design cycle safety is prop of every future op author's care
clones become copies of (Rc, u32)
400+ clone economy mostly evaporates
Storage clones remaining deep copies, get their own fix
top order is free: bw walds id.0 ..= root
nodes made in topo order by cons
dfs in autograd and diamond handling goes away
mem frees deterministically
drop loss, drop graph
no strong count arch
send becomes reachable
one type change of rc to arc because interior refcells are gone
graph under construction owned by 1 thread

costs: tensors from 2 graphs need a check to interact (same rule pytorch engine has)
in place ops need generation or become forbidden (forbid, accumulate_grad is the sanctioned mutation)
biggest change so far, do after 1 and 4 land
precedent: shape micrograd/burn converge on

#### storage: cheap clones, real cache invariant

prevents: deep copy per op econ, unsafe false invariant

```rust
pub enum Storage {
    Host { bytes: Arc<[u8]>, dtype: DType },
    #[cfg(feature = "gpu")]
    Gpu  { buf: Arc<GpuBuffer>, dtype: DType, cache: OnceCell<CachedHost> },
}
```

arc turns every ops data clone snap from O(n) memcpy into refcount bump
mut goes CoW: make_mut cones when strong count > 1
standard Rc make_mut pattern, applied to storage
optim's in place param update only writer in hot path,
usually holds only reference
OnceCell/Lock replaces RefCell opt vec u8 + unsafe as ptr block
write once cache w type-system enforced write-once guarantee
invalidate_cpu_cache either disappears or unsafe block deleted in favor of OnceCell get
either way safety comment stops being false
delete the f64/16 accessors, wrong and unreachable

delete `Deref<Target = [f32]>` and replace w explicit host access:

```rust
impl Storage {
    pub fn as_host_f32(&self) -> Result<&[f32], VoltaError>;   // Err for GPU / non-F32
    pub fn to_host_f32(&self) -> Cow<'_, [f32]>;               // explicit, greppable transfer
}
```

every hidden gpu->cpu sync (invisible behind data.iter today)
every hidden panic (iter mut on gpu, muon bug, h8)
becomes visible result or explicit Cow at call site
grep-ability is a feature
to_host_f32 in a hot loop is a reviewable smell,
deref coercion isn't

#### Loud by default kernels, indexing policy

prevents: silent 0 class
rule making lint table counterproductive was
"never index, always get unwrap or default"
correct policy for kernels separates cases that lint conflates.

```rust
/// For indices that are IN CONSTRUCTION (im2col maps, backward scatters):
/// debug builds assert, release builds clamp-free fetch — but the default is
/// WRONG-BY-DEFAULT loudness, not zero.
#[inline]
#[track_caller]
fn fetch(slice: &[f32], i: usize) -> f32 {
    debug_assert!(i < slice.len(), "index {i} OOB len {} — mapping bug", slice.len());
    slice.get(i).copied().unwrap_or(f32::NAN)   // NaN propagates; zero hides
}
```

Choice of NAN over 0.0 is whole point
index mapping bug in grad kernel produces NaN failing next assert in any test loud
instead of plausible looking number passing a light tol check
debug builds stop at site
combined with define_ops single-source, fewer hand-rolled index maps to get wrong,
shape checked specs
silent wrong number is closed at three independent valves.

pair with: log/tracing crate behind facade
instead of println sites
fallbacks and syncs are observable when asked for, invis otherwise

#### Cfg hygiene and CI that works

prevents: b5 and recurrence
cfg triplication in Direct Conv Grad Fn bw exists because feature gating happens at call sites
gate once at definition sites

```rust
// One file, one definition, two cfg'd bodies. No call site knows about features.
#[cfg(feature = "gpu")]
pub(crate) fn conv_backward_input(...) -> Option<Storage> { /* real impl */ }

#[cfg(not(feature = "gpu"))]
pub(crate) fn conv_backward_input(...) -> Option<Storage> { None }  // 3-line stub
```

callers invoke conv_backward_input unconditionally
not-gpu build compiles stub
sgd's inversion becomes unwritable
one step cpu param, ungated
one gpu dispatch function, gated
audit rule of thumb:
grep for feature outside src/gpu
each hit is a B5 seed
should be 0 after this refactor

CI changes:
step named no features gets no-default-features on both commands
add cargo hack --each feature check: exists for feature combination rot
keep -D warnings but pair with 5.1 and 8 changes
so fix is cheaper than allow

#### test arch that fences bug classes

tests are plentiful, well placed, but blind to config
diffed wrt input
asserted known bugs
never composed ops like models

4 additions fence:

- assert trainable everywhere, regression fence for b1-3. every layer, input-without-grad, every parameter nonzero
- property based grad checks: existing FD checks are one graph each, hand written, and hand-picked tol
  turn invariants into tests w prop

```rust
proptest! {
    #[test]
    fn grads_are_correct_for_random_compositions(seed in any::<u64>(), depth in 2..5usize) {
        // generate a random DAG over {add, mul, matmul, relu, sum_dim, permute}
        // with broadcasting forced at some nodes; FD-check EVERY input of the DAG
    }
}
```

random graph FD check would have caught B4 (unequal rank matmul composes triv in random DAG)
keeps catching whatever op ships next w bad bw, at tol gen controls instead of author

- trybuild UI tests for type-system guarantees
  every pattern in doc promising "misuse doesn't compile"
  gets a tests/compile_fail/*.rs w expected err
  proving prevention instead of hoping
  pattern's val is comp error
  test suite must contain that compile error

- weekly cargo-mutants in CI
  mutates code, fails if tests still pass
  precisely behavior in 'tests that assert the bug'
  lstm test survives every conv tooling choice
  dies when mutant of slice_gate is proven equiv
  test was green infra for a defect
  mut testing can flag it

supp cast:
insta snapshot for stat_dict round-trips
catches key-format drift btwn save and load
seq prefix matching bug in this class
static_assertions day 6 lands (assert_impl_all!(Graph: Send))

#### Sequencing

0-2 alone retire every prev target mapping onto pt1 blocker
arena only genuine arch bet
gets cheaper w each phase before

##### 0: Mechanical, do immediately

Contents: non-exhaustive on pub enums [done], approx dev dep [done], drop statrs [done], CI --no-default-features and cargo hack [done], cfg at defn consolidation [pending — B5], log facade for prints [pending]
Prevents: B5 recurrence, dep bloat, print pollution
Size: day, near 0 risk

##### 1: invariant wall

Contents: Opaque tensor, edge constructor, nograd, assert_trainable across layers
Prevents: B1-3 class, graph surgery
Size: 1 week, mech churn

##### 2: types for dims and ops

Contents: `Shape`/`Axis`/`Strides` (§1.2); specs replace positional args (§1.3); `GradFn` enum (§2.5); `define_ops!` (§5.1); `PadValue` (§2.4); `into_loss` (§2.2)
Prevents: B4 class, fw, bw, kernel drift, H1
Size: 1-2 weeks

##### 3: traits and storage

Contents: Sealed + split traits w/ trybuild (§3); `DevicePolicy` (§2.1); storage `Arc`/`OnceCell`/no-Deref (§7); indexing policy + `fetch` (§8)
Prevents: H3/6/8, silent fallback, clone econ, silent 0
Size: 2 weeks

##### 4: endgame

Contents: Arena graph (§6); `Backend` trait (§3.5); proptest DAG checks (§10.2) if not already in phase 2
Prevents: cycles, lack of Send, memory det
Size: multi-week, do last

#### Explicitly declined patterns

Full device/dtype typstate (2.1)
runtime dynamism is load bearing here
other libs concur

variable/tensor split (2.2):
bounded ScalarLoss version takes real benefit

Const generic rank Tensor:
infects every signature
breaks shape agnostic use code
saves nothing shape newtype doesn't

proc macro op DSL
dep + debug tax unjustified at this op count

ArcMutex today
nothing is Send
area correct route to threading
premature mutexing how bufferpoil guarding state 1 thread can reach

async gpu api
sync blocking design is coherent
make transfers explicit before async

## old notes

main task:
go through pytorch examples (~/src/forks/pytorch-examples/README.md is a good start)
for each:
determine if i can run them in Volta or not right now
if i can
run a side by side comparison on CPU
else
figure out minimal set of changes to make to get them runnable
add tests for each example

most of the basic examples are done

subtask that's probably needed to finish main task:
figure out how to load torch models
how to load other model formats (onnx, candle, ggml, huggingface etc)
go over save/load mechanics, could be unsafe, could accidentally allow loading arbitrary code or smth
add tests for each format

- Full HuggingFace Transformers compatibility
- ONNX support
- Tokenizer integration
- Model hub integration
- Automatic architecture inference from weights
