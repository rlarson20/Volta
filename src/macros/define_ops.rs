/// One table row per op, generating everything that used to be hand-kept in sync
/// (ADDENDUM.md §5.1): the enum, its `ALL` list, the CPU forward, the CPU backward, and
/// the GPU pipeline lookup.
///
/// # Unary rows
///
/// ```ignore
/// define_ops! {
///     /// Doc comment for the enum.
///     unary UnaryOp {
///         /// d(-x)/dx = -1
///         Neg => |x| -x, |g, _x| -g, gpu(neg, neg_backward);
///         /// d(erf x)/dx = 2/√π · exp(-x²)
///         Erf => |x| libm::erff(x), |g, x| g * ..., cpu_only;
///     }
/// }
/// ```
///
/// - `|x| <expr>` is the forward on one `f32`. `|g, x| <expr>` is the backward: `g` is the
///   upstream gradient and `x` the forward input, both `f32`.
/// - `gpu(fwd, bwd)` names two `ComputePipelines` fields, so a kernel that doesn't exist
///   is a compile error rather than a string that silently misses a `match`.
/// - `cpu_only` says outright that the op has no GPU kernel. `gpu_pipelines` returns
///   `None` for it and the caller runs the CPU path.
///
/// There is no default: leaving the backward or the kernel clause off doesn't parse.
macro_rules! define_ops {
    (
        $(#[$emeta:meta])*
        unary $name:ident {
            $(
                $(#[$meta:meta])*
                $variant:ident => |$x:ident| $fwd:expr, |$g:ident, $bx:ident| $bwd:expr,
                    $kind:ident $(($($pipe:ident),*))?;
            )*
        }
    ) => {
        $(#[$emeta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name {
            $(
                $(#[$meta])*
                $variant,
            )*
        }

        impl $name {
            /// Every variant, generated from the same table as the enum.
            pub const ALL: &'static [Self] = &[$(Self::$variant),*];

            /// Apply the op to one element.
            #[must_use]
            pub fn forward(self, input: f32) -> f32 {
                match self {
                    $(Self::$variant => {
                        let $x = input;
                        $fwd
                    })*
                }
            }

            /// Chain rule for one element: `out_grad` times the op's derivative at `input`.
            #[must_use]
            pub fn backward(self, out_grad: f32, input: f32) -> f32 {
                match self {
                    $(Self::$variant => {
                        let $g = out_grad;
                        let $bx = input;
                        $bwd
                    })*
                }
            }

            /// The (forward, backward) GPU pipelines, or `None` for a `cpu_only` op.
            #[cfg(feature = "gpu")]
            #[must_use]
            pub(crate) fn gpu_pipelines(
                self,
                p: &$crate::gpu::ComputePipelines,
            ) -> Option<(&::wgpu::ComputePipeline, &::wgpu::ComputePipeline)> {
                match self {
                    $(Self::$variant => $crate::macros::define_ops!(
                        @unary_pipes p; $kind $(($($pipe),*))?
                    ),)*
                }
            }
        }
    };

    (@unary_pipes $p:ident; cpu_only) => {
        None
    };
    (@unary_pipes $p:ident; gpu($fwd:ident, $bwd:ident)) => {
        Some((&$p.$fwd, &$p.$bwd))
    };
}

pub(crate) use define_ops;
