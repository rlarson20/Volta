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
///
/// # Binary rows
///
/// Two sections, so "has no derivative" is part of the grammar instead of a flag:
///
/// ```ignore
/// define_ops! {
///     binary BinaryOp {
///         differentiable {
///             /// x + y
///             Add => |a, b| a + b, |g, _a, _b| (g, g),
///                 gpu(add, add_backward_a, add_backward_b,
///                     add_broadcast_pass1, add_broadcast_pass2);
///         }
///         non_differentiable {
///             /// x % y
///             Mod => |a, b| a % b, gpu(mod_op);
///         }
///     }
/// }
/// ```
///
/// - The backward returns the per-element pair `(∂L/∂a, ∂L/∂b)` *before* any reduction
///   over broadcast dimensions; the caller does that reduction once for every op.
/// - `gpu(fwd, bwd_a, bwd_b, pass1, pass2)` names the forward, same-shape backward, and
///   two-pass broadcast-backward `ComputePipelines` fields. `gpu(fwd)` is the forward
///   only.
/// - `is_trivial` is true exactly for the `non_differentiable` rows, and their
///   `backward` panics (`edge()` never attaches a trivial grad fn, so it can't run).
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

            /// Whether the op has GPU kernels: `false` for a `cpu_only` row, which is a
            /// gap for the device policy rather than a kernel failure.
            #[must_use]
            pub(crate) const fn has_gpu_kernel(self) -> bool {
                match self {
                    $(Self::$variant => $crate::macros::define_ops!(@unary_has $kind),)*
                }
            }
        }
    };

    (
        $(#[$emeta:meta])*
        binary $name:ident {
            differentiable {
                $(
                    $(#[$dmeta:meta])*
                    $dvariant:ident => |$da:ident, $db:ident| $dfwd:expr,
                        |$dg:ident, $dga:ident, $dgb:ident| $dbwd:expr,
                        gpu($dfp:ident, $dpa:ident, $dpb:ident, $dp1:ident, $dp2:ident);
                )*
            }
            non_differentiable {
                $(
                    $(#[$nmeta:meta])*
                    $nvariant:ident => |$na:ident, $nb:ident| $nfwd:expr, gpu($nfp:ident);
                )*
            }
        }
    ) => {
        $(#[$emeta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name {
            $(
                $(#[$dmeta])*
                $dvariant,
            )*
            $(
                $(#[$nmeta])*
                $nvariant,
            )*
        }

        impl $name {
            /// Every variant, generated from the same table as the enum.
            pub const ALL: &'static [Self] = &[
                $(Self::$dvariant,)*
                $(Self::$nvariant,)*
            ];

            /// Apply the op to one pair of elements.
            #[must_use]
            pub fn forward(self, lhs: f32, rhs: f32) -> f32 {
                match self {
                    $(Self::$dvariant => {
                        let $da = lhs;
                        let $db = rhs;
                        $dfwd
                    })*
                    $(Self::$nvariant => {
                        let $na = lhs;
                        let $nb = rhs;
                        $nfwd
                    })*
                }
            }

            /// Per-element `(∂L/∂lhs, ∂L/∂rhs)` given the upstream gradient.
            ///
            /// # Panics
            /// On a non-differentiable op. `edge()` never attaches a grad fn for one.
            #[must_use]
            pub fn backward(self, out_grad: f32, lhs: f32, rhs: f32) -> (f32, f32) {
                match self {
                    $(Self::$dvariant => {
                        let $dg = out_grad;
                        let $dga = lhs;
                        let $dgb = rhs;
                        $dbwd
                    })*
                    $(Self::$nvariant)|* => {
                        unreachable!("{self:?} is non-differentiable and has no backward")
                    }
                }
            }

            /// Whether the op has no derivative (so `edge()` attaches no grad fn).
            #[must_use]
            pub const fn is_trivial(self) -> bool {
                match self {
                    $(Self::$dvariant)|* => false,
                    $(Self::$nvariant)|* => true,
                }
            }

            /// The forward GPU pipeline. Every binary op has one.
            #[cfg(feature = "gpu")]
            #[must_use]
            pub(crate) fn gpu_forward(
                self,
                p: &$crate::gpu::ComputePipelines,
            ) -> &::wgpu::ComputePipeline {
                match self {
                    $(Self::$dvariant => &p.$dfp,)*
                    $(Self::$nvariant => &p.$nfp,)*
                }
            }

            /// The backward GPU pipelines, or `None` for a non-differentiable op.
            #[cfg(feature = "gpu")]
            #[must_use]
            pub(crate) fn gpu_backward<'p>(
                self,
                p: &'p $crate::gpu::ComputePipelines,
            ) -> Option<$crate::gpu::BinaryBackwardPipelines<'p>> {
                match self {
                    $(Self::$dvariant => Some($crate::gpu::BinaryBackwardPipelines {
                        a: &p.$dpa,
                        b: &p.$dpb,
                        pass1: &p.$dp1,
                        pass2: &p.$dp2,
                    }),)*
                    $(Self::$nvariant)|* => None,
                }
            }
        }
    };

    (@unary_has cpu_only) => {
        false
    };
    (@unary_has gpu) => {
        true
    };
    (@unary_pipes $p:ident; cpu_only) => {
        None
    };
    (@unary_pipes $p:ident; gpu($fwd:ident, $bwd:ident)) => {
        Some((&$p.$fwd, &$p.$bwd))
    };
}

pub(crate) use define_ops;
