//! `DType::promote` against `torch.result_type` on all 64 pairs of Volta dtypes
//! (`tests/fixtures/torch/dtypes.safetensors`).

#[path = "torch_ref/mod.rs"]
mod torch_ref;

use torch_ref::Fixture;
use volta::DType;

/// Volta's dtypes in discriminant order, the order the fixture's rows and columns use
/// (`VOLTA_DTYPES` in `scripts/torch_ref/gen_fixtures.py`).
const ORDER: [DType; 8] = [
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::Bool,
];

/// The fixture stores dtypes by discriminant; a reordered `DType` would misread it.
#[test]
fn order_matches_the_discriminants_the_fixture_encodes() {
    for (code, dtype) in ORDER.iter().enumerate() {
        assert_eq!(usize::from(*dtype as u8), code, "{dtype}");
    }
}

/// Every pair, collected, so one run lists all disagreements.
#[test]
fn promote_matches_torch_result_type() {
    let f = Fixture::load("dtypes");
    let (codes, shape) = f.get("promotion");
    assert_eq!(shape, vec![ORDER.len(), ORDER.len()]);

    let mut wrong = Vec::new();
    for (i, a) in ORDER.iter().enumerate() {
        for (j, b) in ORDER.iter().enumerate() {
            let code = codes[i * ORDER.len() + j];
            assert!(
                code.fract() == 0.0 && (0.0..8.0).contains(&code),
                "fixture code {code} for ({a}, {b}) is not a dtype index"
            );
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "checked above to be an integer in 0..8"
            )]
            let torch = ORDER[code as usize];
            let volta = DType::promote(*a, *b);
            if volta != torch {
                wrong.push(format!("({a}, {b}): volta {volta}, torch {torch}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "promotion differs from torch:\n{}",
        wrong.join("\n")
    );
}
