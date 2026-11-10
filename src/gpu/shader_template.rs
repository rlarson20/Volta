//! One WGSL source per shader, expanded per element type (ADDENDUM §11 P8).
//!
//! A shader declares each data buffer as `array<ELEM>` (or `array<ELEM2>`, a second
//! element type, for casts) and never indexes it directly: it reads with `ld_NAME(i)`,
//! writes with `st_NAME(i, v)` and bounds with `len_NAME()`, all in f32 (`i: u32`,
//! `v: f32`). [`expand`] replaces each declaration and appends those helpers for the
//! chosen [`Elem`]:
//!
//! - [`Elem::F32`]: `array<f32>`, plain indexing.
//! - [`Elem::F16`]: `array<f16>` (wgpu `SHADER_F16`); loads widen exactly, stores round
//!   to nearest-even through the integer rounding below, then convert the (now exactly
//!   representable) value.
//! - [`Elem::F16Packed`] and [`Elem::Bf16Packed`]: two 16-bit values per `u32` word, low
//!   half first, which is the same byte layout as the CPU's `HostBuf::F16`/`BF16` (and as
//!   `array<f16>`), so uploads and downloads are plain byte copies. WGSL has no 16-bit
//!   integer type, so this is the only spec-safe 16-bit storage without `SHADER_F16`: a
//!   store must not clobber the other half, which another thread may be writing, so it is
//!   an `atomicAnd` that clears this half and an `atomicOr` that sets it. This is
//!   PyTorch's `c10::BFloat16` emulation (16-bit storage, f32 compute, round-to-nearest-even
//!   store, NaN as `0x7FC0`) in WGSL terms.
//!
//! Every kernel therefore computes in f32 and rounds once per stored element, as the CPU
//! does for half types (P4). `tests` validates every shader under every element type with
//! naga, without a device.

use std::fmt::Write as _;

/// The storage element type a shader is expanded for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Elem {
    /// `f32`.
    F32,
    /// Native `f16` (`enable f16;`).
    F16,
    /// f16 bits packed two per `u32` (adapters without `SHADER_F16`).
    F16Packed,
    /// bf16 bits packed two per `u32` (every adapter: WGSL has no bf16).
    Bf16Packed,
}

impl Elem {
    /// Every element type, for validation.
    #[cfg(test)]
    pub(crate) const ALL: [Self; 4] = [Self::F32, Self::F16, Self::F16Packed, Self::Bf16Packed];
}

/// A data buffer declaration found in a shader.
struct Decl {
    /// The buffer's name.
    name: String,
    /// `read_write` (else `read`).
    writable: bool,
    /// Its element type in this expansion.
    elem: Elem,
}

/// `source` with every `array<ELEM>` buffer as `elem` and every `array<ELEM2>` buffer as
/// `elem2`, plus their `ld_`/`st_`/`len_` helpers and the conversions they need.
///
/// # Panics
/// If a line declares an `ELEM` buffer in a form this expander does not parse (a
/// shader-source bug, caught by the validation test).
pub(crate) fn expand(source: &str, elem: Elem, elem2: Elem) -> String {
    let mut body = String::with_capacity(source.len() + 2048);
    let mut decls = Vec::new();
    for line in source.lines() {
        if let Some(d) = parse_decl(line, elem, elem2) {
            let prefix = line
                .split("var<")
                .next()
                .expect("a declaration has a `var<`");
            let access = if d.writable { "read_write" } else { "read" };
            let ty = storage_type(d.elem, d.writable);
            let _ = writeln!(body, "{prefix}var<storage, {access}> {}: {ty};", d.name);
            decls.push(d);
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }

    let mut out = String::new();
    if decls.iter().any(|d| d.elem == Elem::F16) {
        out.push_str("enable f16;\n");
    }
    out.push_str(&body);
    out.push_str(CONVERSIONS);
    for d in &decls {
        helpers(&mut out, d);
    }
    out
}

/// The declaration on `line`, if it declares an `array<ELEM>` or `array<ELEM2>` buffer.
fn parse_decl(line: &str, elem: Elem, elem2: Elem) -> Option<Decl> {
    let code = line.split("//").next().unwrap_or("");
    let which = if code.contains("array<ELEM2>") {
        elem2
    } else if code.contains("array<ELEM>") {
        elem
    } else {
        return None;
    };
    let after = code.split("var<storage,").nth(1).unwrap_or_else(|| {
        panic!("shader template: an ELEM buffer must be `var<storage, ..>`: {line}")
    });
    let (access, rest) = after
        .split_once('>')
        .unwrap_or_else(|| panic!("shader template: unclosed var<..>: {line}"));
    let name = rest
        .split(':')
        .next()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| panic!("shader template: no buffer name: {line}"));
    Some(Decl {
        name: name.to_string(),
        writable: access.trim() == "read_write",
        elem: which,
    })
}

/// The WGSL type of a buffer of `elem`.
const fn storage_type(elem: Elem, writable: bool) -> &'static str {
    match (elem, writable) {
        (Elem::F32, _) => "array<f32>",
        (Elem::F16, _) => "array<f16>",
        (Elem::F16Packed | Elem::Bf16Packed, false) => "array<u32>",
        (Elem::F16Packed | Elem::Bf16Packed, true) => "array<atomic<u32>>",
    }
}

/// Append `ld_`, `len_` and (for writable buffers) `st_` for `d`.
fn helpers(out: &mut String, d: &Decl) {
    let n = &d.name;
    let _ = match d.elem {
        Elem::F32 => write!(
            out,
            "fn ld_{n}(i: u32) -> f32 {{ return {n}[i]; }}\n\
             fn len_{n}() -> u32 {{ return arrayLength(&{n}); }}\n"
        ),
        Elem::F16 => write!(
            out,
            "fn ld_{n}(i: u32) -> f32 {{ return f32({n}[i]); }}\n\
             fn len_{n}() -> u32 {{ return arrayLength(&{n}); }}\n"
        ),
        Elem::F16Packed | Elem::Bf16Packed => {
            let word = if d.writable {
                format!("atomicLoad(&{n}[i >> 1u])")
            } else {
                format!("{n}[i >> 1u]")
            };
            let decode = if d.elem == Elem::F16Packed {
                "f16_bits_to_f32"
            } else {
                "bf16_bits_to_f32"
            };
            write!(
                out,
                "fn ld_{n}(i: u32) -> f32 {{ return {decode}(({word} >> ((i & 1u) * 16u)) & 0xffffu); }}\n\
                 fn len_{n}() -> u32 {{ return arrayLength(&{n}) * 2u; }}\n"
            )
        }
    };
    if !d.writable {
        return;
    }
    let _ = match d.elem {
        Elem::F32 => writeln!(out, "fn st_{n}(i: u32, v: f32) {{ {n}[i] = v; }}"),
        Elem::F16 => writeln!(
            out,
            // Round in integer bits, then convert the exactly representable result: the
            // `f16()` conversion of a value that is already an f16 is exact.
            "fn st_{n}(i: u32, v: f32) {{ {n}[i] = f16(f16_bits_to_f32(f32_to_f16_bits(v))); }}"
        ),
        Elem::F16Packed | Elem::Bf16Packed => {
            let encode = if d.elem == Elem::F16Packed {
                "f32_to_f16_bits"
            } else {
                "f32_to_bf16_bits"
            };
            writeln!(
                out,
                "fn st_{n}(i: u32, v: f32) {{\n\
                 \x20   let shift = (i & 1u) * 16u;\n\
                 \x20   atomicAnd(&{n}[i >> 1u], ~(0xffffu << shift));\n\
                 \x20   atomicOr(&{n}[i >> 1u], {encode}(v) << shift);\n\
                 }}"
            )
        }
    };
}

/// Bit-exact conversions between f32 and the 16-bit formats, in integer arithmetic so
/// rounding (to nearest, ties to even), overflow (to inf) and NaN don't depend on the
/// GPU's float conversion, which WGSL leaves partly implementation-defined.
const CONVERSIONS: &str = r"
// f32 -> IEEE half bits, round to nearest even; overflow to inf; NaN to 0x7e00.
fn f32_to_f16_bits(v: f32) -> u32 {
    let x = bitcast<u32>(v);
    let sign = (x >> 16u) & 0x8000u;
    let ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) { return sign | 0x7e00u; }
    if (ax >= 0x477ff000u) { return sign | 0x7c00u; }
    if (ax < 0x38800000u) {
        if (ax < 0x33000000u) { return sign; }
        let e = ax >> 23u;
        let m = (ax & 0x7fffffu) | 0x800000u;
        let shift = 126u - e;
        let q = m >> shift;
        let rem = m & ((1u << shift) - 1u);
        let half = 1u << (shift - 1u);
        var r = q;
        if (rem > half || (rem == half && (q & 1u) == 1u)) { r = q + 1u; }
        return sign | r;
    }
    let base = ax - 0x38000000u;
    let q = base >> 13u;
    let rem = base & 0x1fffu;
    var r = q;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u) == 1u)) { r = q + 1u; }
    return sign | r;
}

// IEEE half bits -> f32, exactly (subnormals included).
fn f16_bits_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let e = (h >> 10u) & 0x1fu;
    let m = h & 0x3ffu;
    if (e == 0u) {
        let mag = f32(m) * 5.9604644775390625e-8;
        return select(mag, -mag, sign != 0u);
    }
    if (e == 31u) { return bitcast<f32>(sign | 0x7f800000u | (m << 13u)); }
    return bitcast<f32>(sign | ((e + 112u) << 23u) | (m << 13u));
}

// f32 -> bf16 bits, round to nearest even; NaN to 0x7fc0 (PyTorch's c10::BFloat16).
fn f32_to_bf16_bits(v: f32) -> u32 {
    let x = bitcast<u32>(v);
    if ((x & 0x7fffffffu) > 0x7f800000u) { return 0x7fc0u; }
    return (x + 0x7fffu + ((x >> 16u) & 1u)) >> 16u;
}

// The flat invocation index of a 256-wide 1-D kernel dispatched as a 2-D grid
// (`kernels::dispatch_flat`): wgpu caps each dispatch dimension at 65535 workgroups, so
// a launch over more than 65535 * 256 elements spreads its rows over `y`. Returned as a
// vec3 so a kernel can keep reading `global_id.x`.
fn flat_global_id(gid: vec3<u32>, num_wg: vec3<u32>) -> vec3<u32> {
    return vec3<u32>(gid.x + gid.y * num_wg.x * 256u, 0u, 0u);
}

// bf16 bits -> f32, exactly.
fn bf16_bits_to_f32(b: u32) -> f32 {
    return bitcast<f32>(b << 16u);
}

// tanh that stays finite for large |x| (S24). Metal evaluates tanh as
// (e^2x - 1) / (e^2x + 1) under fast math, which returns 0 for 43.67 < x < 44.36 (the
// denominator passes 2^126, where fast division flushes to 0) and NaN above that
// (e^2x = inf). tanh rounds to exactly 1.0 in f32 for |x| > 9.01, so clamping to +/-15
// changes no result and keeps e^2x near 1e13. Metal's clamp(NaN) returns -15, which
// would turn a NaN into -1, so a NaN is detected from its bits (fast math may fold
// x != x away) and passed through.
fn tanh_sat(x: f32) -> f32 {
    let is_nan = (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
    return select(tanh(clamp(x, -15.0, 15.0)), x, is_nan);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shader file, with the element types its `ELEM2` buffers take (`None`: it has
    /// none).
    pub(crate) const SHADERS: &[(&str, &str)] = &[
        ("amp_unscale", include_str!("shaders/amp_unscale.wgsl")),
        (
            "binary_backward",
            include_str!("shaders/binary_backward.wgsl"),
        ),
        (
            "binary_backward_safe",
            include_str!("shaders/binary_backward_safe.wgsl"),
        ),
        ("cast", include_str!("shaders/cast.wgsl")),
        ("col2im", include_str!("shaders/col2im.wgsl")),
        ("compare", include_str!("shaders/compare.wgsl")),
        (
            "conv_backward_input",
            include_str!("shaders/conv_backward_input.wgsl"),
        ),
        (
            "conv_backward_weight",
            include_str!("shaders/conv_backward_weight.wgsl"),
        ),
        ("direct_conv", include_str!("shaders/direct_conv.wgsl")),
        ("elementwise", include_str!("shaders/elementwise.wgsl")),
        ("igemm", include_str!("shaders/igemm.wgsl")),
        (
            "igemm_backward_input",
            include_str!("shaders/igemm_backward_input.wgsl"),
        ),
        (
            "igemm_backward_weight",
            include_str!("shaders/igemm_backward_weight.wgsl"),
        ),
        ("im2col", include_str!("shaders/im2col.wgsl")),
        ("matmul", include_str!("shaders/matmul.wgsl")),
        (
            "matmul_backward",
            include_str!("shaders/matmul_backward.wgsl"),
        ),
        ("movement", include_str!("shaders/movement.wgsl")),
        (
            "movement_backward",
            include_str!("shaders/movement_backward.wgsl"),
        ),
        ("muon", include_str!("shaders/muon.wgsl")),
        (
            "optimizer_step",
            include_str!("shaders/optimizer_step.wgsl"),
        ),
        ("reduce", include_str!("shaders/reduce.wgsl")),
        (
            "reduce_backward",
            include_str!("shaders/reduce_backward.wgsl"),
        ),
        ("unary", include_str!("shaders/unary.wgsl")),
        (
            "unary_backward",
            include_str!("shaders/unary_backward.wgsl"),
        ),
        ("where_select", include_str!("shaders/where_select.wgsl")),
    ];

    /// Parse and validate `src` with naga, allowing f16.
    fn validate(label: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{label}: parse error:\n{}", e.emit_to_string(src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::SHADER_FLOAT16,
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: invalid:\n{}", e.emit_to_string(src)));
    }

    /// Every shader expands to valid WGSL under every pair of element types.
    #[test]
    fn every_shader_validates_under_every_element_type() {
        for (name, src) in SHADERS {
            for elem in Elem::ALL {
                for elem2 in Elem::ALL {
                    validate(
                        &format!("{name} ({elem:?}, {elem2:?})"),
                        &expand(src, elem, elem2),
                    );
                }
            }
        }
    }

    /// No shader indexes a data buffer directly or bounds it with `arrayLength`: every
    /// access goes through the helpers, so a packed expansion can't read the wrong half.
    #[test]
    fn shaders_access_data_buffers_only_through_helpers() {
        for (name, src) in SHADERS {
            let mut decls = Vec::new();
            for line in src.lines() {
                if let Some(d) = parse_decl(line, Elem::F32, Elem::F32) {
                    decls.push(d.name);
                }
            }
            for (no, line) in src.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                if code.contains("array<ELEM") {
                    continue;
                }
                for d in &decls {
                    let bad = [format!("{d}["), format!("&{d})"), format!("&{d}[")];
                    for b in bad {
                        let hit = code.match_indices(&b).any(|(at, _)| {
                            at == 0
                                || !code[..at]
                                    .chars()
                                    .next_back()
                                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
                        });
                        assert!(
                            !hit,
                            "{name}.wgsl:{}: direct access to `{d}`: {line}",
                            no + 1
                        );
                    }
                }
            }
        }
    }

    /// The f32 expansion of a declaration is the plain f32 declaration.
    #[test]
    fn f32_expansion_keeps_plain_arrays() {
        let src = "@group(0) @binding(0) var<storage, read> a: array<ELEM>;\n";
        let out = expand(src, Elem::F32, Elem::F32);
        assert!(out.contains("var<storage, read> a: array<f32>;"), "{out}");
        assert!(
            out.contains("fn ld_a(i: u32) -> f32 { return a[i]; }"),
            "{out}"
        );
        assert!(!out.contains("enable f16"));
    }
}
