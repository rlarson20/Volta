# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "safetensors", "numpy"]
#
# # The CPU-only torch build: PyPI's Linux wheel pulls in several GB of CUDA libraries.
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Generate PyTorch reference fixtures for the Volta `torch_ref_*` tests.

Each group writes `tests/fixtures/torch/<group>.safetensors` with f32 tensors:
inputs, forward outputs, and gradients. Run it with `just torch-fixtures`; commit the
result. The torch version and seed are stored in the file metadata.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "torch"
SEED = 1234
INF = float("inf")
NAN = float("nan")


def leaf(values, shape=None):
    t = torch.tensor(values, dtype=torch.float32)
    if shape is not None:
        t = t.reshape(shape)
    return t.clone().requires_grad_(True)


def save(group, tensors):
    OUT.mkdir(parents=True, exist_ok=True)
    # One metadata key: safetensors writes the metadata map in hash order, so several keys
    # would make the file bytes differ between runs.
    meta = {"info": f"torch={torch.__version__};seed={SEED};group={group}"}
    save_file({k: v.detach().contiguous().float() for k, v in tensors.items()}, OUT / f"{group}.safetensors", metadata=meta)
    print(f"wrote {group}: {len(tensors)} tensors")


def grad_of(out, *inputs, weights=None):
    """Gradient of sum(out * weights) w.r.t. each input (weights default to ones)."""
    w = torch.ones_like(out) if weights is None else weights
    return torch.autograd.grad((out * w).sum(), inputs, allow_unused=True)


def group_ops():
    """Elementwise ops, ties, NaN, remainder, and BCE edge cases."""
    torch.manual_seed(SEED)
    t = {}

    # S4: max_elem on a tie. torch.maximum splits the gradient evenly.
    a, b = leaf([3.0, 1.0, 2.0]), leaf([3.0, 2.0, 2.0])
    out = torch.maximum(a, b)
    ga, gb = grad_of(out, a, b)
    t.update({"max_elem.a": a, "max_elem.b": b, "max_elem.out": out, "max_elem.grad_a": ga, "max_elem.grad_b": gb})

    # S4: NaN propagates through maximum.
    # A NaN operand sends the whole gradient to both sides.
    a, b = leaf([NAN, 1.0]), leaf([1.0, NAN])
    out = torch.maximum(a, b)
    ga, gb = grad_of(out, a, b)
    t.update({"max_elem_nan.a": a, "max_elem_nan.b": b, "max_elem_nan.out": out, "max_elem_nan.grad_a": ga, "max_elem_nan.grad_b": gb})

    # S4: full max() with a tie splits the gradient evenly.
    x = leaf([2.0, 5.0, 5.0, 1.0])
    out = x.max()
    (gx,) = grad_of(out, x)
    t.update({"max_all_tie.x": x, "max_all_tie.out": out.reshape(1), "max_all_tie.grad": gx})

    # S4: full max() with NaN returns NaN.
    x = leaf([1.0, NAN, 3.0])
    t.update({"max_all_nan.x": x, "max_all_nan.out": x.max().reshape(1)})

    # S4: two NaNs in a full max(): the output is NaN and the gradient splits over the NaNs.
    x = leaf([NAN, 2.0, NAN, 1.0])
    out = x.max()
    (gx,) = grad_of(out, x)
    t.update({"max_all_nan2.x": x, "max_all_nan2.out": out.reshape(1), "max_all_nan2.grad": gx})

    # S4: relu(NaN) is NaN.
    x = leaf([NAN, -1.0, 2.0])
    t.update({"relu_nan.x": x, "relu_nan.out": torch.relu(x)})

    # S5: remainder (floored, sign of divisor) and its gradient.
    a, b = leaf([-1.0, 5.5, -7.0, 7.0]), leaf([3.0, 2.0, 2.0, -3.0])
    out = torch.remainder(a, b)
    ga, gb = grad_of(out, a, b)
    t.update({"remainder.a": a, "remainder.b": b, "remainder.out": out, "remainder.grad_a": ga, "remainder.grad_b": gb})

    # S2: max_dim on a row with no winner (all -inf); gradient must stay in its own row.
    x = leaf([[1.0, 2.0], [-INF, -INF]])
    out = x.max(dim=1).values
    (gx,) = grad_of(out, x, weights=torch.tensor([1.0, 10.0]))
    t.update({"max_dim_neginf.x": x, "max_dim_neginf.out": out, "max_dim_neginf.grad": gx})

    # S2: max_dim with NaN in a row returns NaN.
    x = leaf([[NAN, 3.0], [1.0, 2.0]])
    t.update({"max_dim_nan.x": x, "max_dim_nan.out": x.max(dim=1).values})

    # max(dim) on a tie: one winner (the first index) gets the whole gradient.
    x = leaf([[2.0, 5.0, 5.0], [7.0, 7.0, 1.0]])
    out = x.max(dim=1).values
    (gx,) = grad_of(out, x, weights=torch.tensor([1.0, 10.0]))
    t.update({"max_dim_tie.x": x, "max_dim_tie.out": out, "max_dim_tie.grad": gx})

    # max(dim) with two NaNs in a row: NaN out, the gradient goes to one NaN.
    x = leaf([[NAN, 3.0, NAN], [1.0, 2.0, 3.0]])
    out = x.max(dim=1).values
    (gx,) = grad_of(out, x, weights=torch.tensor([1.0, 10.0]))
    t.update({"max_dim_nan2.x": x, "max_dim_nan2.out": out, "max_dim_nan2.grad": gx})

    # S2: softmax of a fully masked (-inf) row is NaN in torch, and must not touch row 0.
    x = leaf([[1.0, 2.0], [-INF, -INF]])
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = torch.softmax(x, dim=1)
    (gx,) = grad_of(out, x, weights=w)
    t.update({"softmax_masked.x": x, "softmax_masked.w": w, "softmax_masked.out": out, "softmax_masked.grad": gx})

    # S7: BCE at the clamp (log clamped to >= -100).
    p, y = leaf([0.0, 1.0, 0.0, 1.0, 0.3]), torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    out = torch.nn.functional.binary_cross_entropy(p, y, reduction="mean")
    (gp,) = grad_of(out, p)
    t.update({"bce.pred": p, "bce.target": y, "bce.out": out.reshape(1), "bce.grad": gp})

    # S8: cross entropy where a non-target class has logit -inf.
    z = leaf([[1.0, -INF, 2.0], [0.5, 0.2, -INF]])
    cls = torch.tensor([0, 1])
    out = torch.nn.functional.cross_entropy(z, cls)
    (gz,) = grad_of(out, z)
    # Probability targets multiply (0 * -inf), so torch's answer there is NaN.
    probs = torch.nn.functional.one_hot(cls, 3).float()
    pout = torch.nn.functional.cross_entropy(z.detach(), probs)
    t.update({
        "ce_neginf.logits": z,
        "ce_neginf.classes": cls.float(),
        "ce_neginf.out": out.reshape(1),
        "ce_neginf.grad": gz,
        "ce_neginf.probs": probs,
        "ce_neginf.probs_out": pout.reshape(1),
    })

    save("ops", t)


def group_reductions():
    """Sums, means, softmax; large-sum accuracy is checked in Rust against 2^25."""
    torch.manual_seed(SEED)
    t = {}
    x = leaf(torch.randn(3, 4, 5).tolist())
    for dim in range(3):
        s = x.sum(dim=dim)
        m = x.mean(dim=dim, keepdim=True)
        w = torch.randn_like(s)
        (gs,) = grad_of(s, x, weights=w)
        t[f"sum_dim{dim}.out"] = s
        t[f"sum_dim{dim}.w"] = w
        t[f"sum_dim{dim}.grad"] = gs
        t[f"mean_dim{dim}.out"] = m
        sm = torch.softmax(x, dim=dim)
        ls = torch.log_softmax(x, dim=dim)
        w2 = torch.randn_like(sm)
        (gsm,) = grad_of(sm, x, weights=w2)
        (gls,) = grad_of(ls, x, weights=w2)
        t[f"softmax_dim{dim}.out"] = sm
        t[f"softmax_dim{dim}.w"] = w2
        t[f"softmax_dim{dim}.grad"] = gsm
        t[f"log_softmax_dim{dim}.out"] = ls
        t[f"log_softmax_dim{dim}.grad"] = gls
    t["x"] = x
    save("reductions", t)


def trajectory(t, name, make_opt, shapes, grad_steps, steps=5):
    """Run `steps` optimizer steps with fixed gradients and record each parameter after each.

    `grad_steps[i]` lists the steps (0-based) at which parameter `i` receives a gradient.
    """
    torch.manual_seed(SEED)
    params = [torch.randn(*sh, requires_grad=True) for sh in shapes]
    for i, p in enumerate(params):
        t[f"{name}.p{i}.init"] = p.detach().clone()
    opt = make_opt(params)
    for k in range(steps):
        for i, p in enumerate(params):
            if k in grad_steps[i]:
                g = torch.randn(*shapes[i])
                p.grad = g.clone()
                t[f"{name}.p{i}.g{k}"] = g
            else:
                p.grad = None
        opt.step()
        for i, p in enumerate(params):
            t[f"{name}.p{i}.after{k}"] = p.detach().clone()


def group_optim():
    """SGD, Adam, and Muon trajectories under fixed gradients (S17)."""
    t = {}
    every = set(range(5))
    trajectory(t, "sgd_plain", lambda ps: torch.optim.SGD(ps, lr=0.1), [(2, 3)], [every])
    trajectory(t, "sgd_mom_wd", lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9, weight_decay=0.01), [(2, 3)], [every])
    trajectory(t, "adam", lambda ps: torch.optim.Adam(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01), [(2, 3)], [every])
    # Parameter 1 first receives a gradient at step 2; torch counts steps per parameter.
    trajectory(t, "adam_late", lambda ps: torch.optim.Adam(ps, lr=0.01), [(4,), (4,)], [every, {2, 3, 4}])
    muon_trajectory(t, "muon_wide", (2, 3))
    muon_trajectory(t, "muon_tall", (3, 2))
    save("optim", t)


def muon_f32_step(p, g, buf, lr=0.02, momentum=0.95, ns_steps=5, eps=1e-7):
    """torch.optim.Muon's step (_single_tensor_muon) with Newton-Schulz in float32.

    torch runs Newton-Schulz in bfloat16, which is about 1% off a float32 implementation,
    so it cannot be the expected value for Volta's float32 Muon. This transcription of the
    same iteration is checked against torch.optim.Muon below.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    buf.lerp_(g, 1 - momentum)
    x = g.lerp(buf, momentum)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(ns_steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if tall:
        x = x.T
    scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
    p.add_(x, alpha=-lr * scale)


def muon_trajectory(t, name, shape, steps=5):
    torch.manual_seed(SEED)
    p_ref = torch.randn(*shape)
    p_torch = p_ref.clone().requires_grad_(True)
    t[f"{name}.p0.init"] = p_ref.clone()
    opt = torch.optim.Muon([p_torch], lr=0.02, weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5)
    buf = torch.zeros(*shape)
    for k in range(steps):
        g = torch.randn(*shape)
        t[f"{name}.p0.g{k}"] = g
        p_torch.grad = g.clone()
        opt.step()
        muon_f32_step(p_ref, g, buf)
        t[f"{name}.p0.after{k}"] = p_ref.clone()
        # The float32 transcription must agree with torch.optim.Muon to bfloat16 precision.
        drift = (p_ref - p_torch.detach()).abs().max().item()
        assert drift < 5e-3, f"{name} step {k}: f32 reference drifts {drift} from torch.optim.Muon"


def case(t, name, out, inputs, seed_w=True, w=None):
    """Record `out`, a weighting `w`, and the gradient of sum(out * w) per input.

    `inputs` maps a short name to a leaf tensor already stored by the caller. `w` defaults
    to random draws; pass one to leave the RNG stream (and every later case) untouched.
    """
    if w is None:
        w = torch.randn_like(out)
    names = list(inputs)
    grads = torch.autograd.grad((out * w).sum(), [inputs[n] for n in names], allow_unused=True)
    t[f"{name}.out"] = out
    t[f"{name}.w"] = w
    for n, g in zip(names, grads):
        t[f"{name}.grad_{n}"] = torch.zeros_like(inputs[n]) if g is None else g


def group_layers():
    """Layers and matmul/movement ops with explicit weights, forward and backward."""
    torch.manual_seed(SEED)
    F = torch.nn.functional
    t = {}

    def rl(*shape):
        return torch.randn(*shape).requires_grad_(True)

    # Linear: torch weight is [out, in]; Volta stores [in, out].
    x, w, b = rl(4, 2), rl(3, 2), rl(3)
    t.update({"linear.x": x, "linear.weight": w, "linear.bias": b})
    case(t, "linear", F.linear(x, w, b), {"x": x, "weight": w, "bias": b})

    # LayerNorm over the last dim.
    x, g, b = rl(3, 4), rl(4), rl(4)
    t.update({"layernorm.x": x, "layernorm.gamma": g, "layernorm.beta": b})
    case(t, "layernorm", F.layer_norm(x, (4,), g, b, 1e-5), {"x": x, "gamma": g, "beta": b})

    # GELU (exact, erf form).
    x = rl(2, 5)
    t["gelu.x"] = x
    case(t, "gelu", F.gelu(x), {"x": x})

    # Conv2d with stride 2, padding 1.
    x, w, b = rl(2, 2, 5, 5), rl(3, 2, 3, 3), rl(3)
    t.update({"conv.x": x, "conv.weight": w, "conv.bias": b})
    case(t, "conv", F.conv2d(x, w, b, stride=2, padding=1), {"x": x, "weight": w, "bias": b})

    # ConvTranspose2d, stride 2, padding 1; weight is [in, out, k, k] in both libraries.
    x, w, b = rl(2, 2, 3, 3), rl(2, 3, 3, 3), rl(3)
    t.update({"convt.x": x, "convt.weight": w, "convt.bias": b})
    case(t, "convt", F.conv_transpose2d(x, w, b, stride=2, padding=1), {"x": x, "weight": w, "bias": b})

    # MaxPool2d 2x2 stride 2.
    x = rl(2, 3, 4, 4)
    t["maxpool.x"] = x
    case(t, "maxpool", F.max_pool2d(x, 2, 2), {"x": x})

    # S3: a channel that is entirely -inf. Its windows have no winner; torch routes the
    # gradient to the first element of each window, never to another channel.
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[-INF] * 2, [-INF] * 2]]], requires_grad=True)
    t["maxpool_neginf.x"] = x
    out = F.max_pool2d(x, 2, 2)
    w = torch.tensor([1.0, 10.0]).reshape(1, 2, 1, 1)
    t["maxpool_neginf.out"] = out
    t["maxpool_neginf.w"] = w
    t["maxpool_neginf.grad_x"] = torch.autograd.grad((out * w).sum(), x)[0]

    # S3: NaN in a window propagates.
    x = torch.tensor([[[[1.0, NAN], [2.0, 3.0]]]])
    t["maxpool_nan.x"] = x
    t["maxpool_nan.out"] = F.max_pool2d(x, 2, 2)

    # A tie in a window: the first element in scan order wins the whole gradient.
    x = torch.tensor([[[[5.0, 5.0], [1.0, 5.0]]]], requires_grad=True)
    t["maxpool_tie.x"] = x
    case(t, "maxpool_tie", F.max_pool2d(x, 2, 2), {"x": x}, w=torch.tensor([3.0]).reshape(1, 1, 1, 1))

    # Two NaNs in a window: NaN out, and torch picks one NaN for the gradient.
    x = torch.tensor([[[[NAN, 1.0], [NAN, 3.0]]]], requires_grad=True)
    t["maxpool_nan2.x"] = x
    case(t, "maxpool_nan2", F.max_pool2d(x, 2, 2), {"x": x}, w=torch.tensor([3.0]).reshape(1, 1, 1, 1))

    # S3 with padding: a channel of -inf has windows that mix padding and real -inf.
    x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[-INF] * 3] * 3]], requires_grad=True)
    t["maxpool_neginf_pad.x"] = x
    out = F.max_pool2d(x, 2, 2, padding=1)
    w = torch.arange(1.0, 9.0).reshape(1, 2, 2, 2)
    t["maxpool_neginf_pad.out"] = out
    t["maxpool_neginf_pad.w"] = w
    t["maxpool_neginf_pad.grad_x"] = torch.autograd.grad((out * w).sum(), x)[0]

    # BatchNorm2d, training mode: output, gradients, and the running statistics it leaves.
    x, g, b = rl(3, 2, 2, 2), rl(2), rl(2)
    rm, rv = torch.zeros(2), torch.ones(2)
    t.update({"bn.x": x, "bn.gamma": g, "bn.beta": b})
    out = F.batch_norm(x, rm, rv, g, b, training=True, momentum=0.1, eps=1e-5)
    case(t, "bn", out, {"x": x, "gamma": g, "beta": b})
    t["bn.running_mean"], t["bn.running_var"] = rm.clone(), rv.clone()
    xe = torch.randn(3, 2, 2, 2)
    t["bn.x_eval"] = xe
    t["bn.out_eval"] = F.batch_norm(xe, rm, rv, g.detach(), b.detach(), training=False, eps=1e-5)

    # Matmul variants.
    a, c = rl(3, 4), rl(4, 2)
    t.update({"mm22.a": a, "mm22.b": c})
    case(t, "mm22", a @ c, {"a": a, "b": c})
    a, c = rl(2, 3, 4), rl(2, 4, 5)
    t.update({"mm33.a": a, "mm33.b": c})
    case(t, "mm33", a @ c, {"a": a, "b": c})
    a, c = rl(2, 3, 4), rl(4, 5)
    t.update({"mm32.a": a, "mm32.b": c})
    case(t, "mm32", a @ c, {"a": a, "b": c})
    a, c = rl(2, 1, 3, 4), rl(1, 3, 4, 5)
    t.update({"mm44.a": a, "mm44.b": c})
    case(t, "mm44", a @ c, {"a": a, "b": c})

    # Movement ops.
    x = rl(2, 3, 4)
    t["permute.x"] = x
    case(t, "permute", x.permute(1, 2, 0), {"x": x})
    x = rl(1, 8, 2, 3)
    t["pixelshuffle.x"] = x
    case(t, "pixelshuffle", F.pixel_shuffle(x, 2), {"x": x})

    save("layers", t)


def module_case(t, name, module, x, out, split_in_proj):
    """Record a module's weights, `out`, a random weighting `w`, and the gradients of
    sum(out * w) for `x` and every parameter, keyed by the parameter's torch name.

    `split_in_proj` names the fused `in_proj_weight` parameters (`[3E, E]`) to store as
    separate `q`/`k`/`v` weights and gradients, the layout Volta uses. Volta's q/k/v
    projections have no bias, so the caller zeroes `in_proj_bias` and it is not stored.
    """
    params = [(n, p) for n, p in module.named_parameters() if not n.endswith("in_proj_bias")]
    w = torch.randn_like(out)
    grads = torch.autograd.grad((out * w).sum(), [x] + [p for _, p in params])
    t[f"{name}.x"] = x
    t[f"{name}.out"] = out
    t[f"{name}.w"] = w
    t[f"{name}.grad_x"] = grads[0]
    for (n, p), g in zip(params, grads[1:]):
        if n in split_in_proj:
            stem = n.removesuffix("in_proj_weight")
            for part, pv, gv in zip("qkv", p.detach().chunk(3), g.chunk(3)):
                t[f"{name}.{stem}{part}_proj.weight"] = pv
                t[f"{name}.grad_{stem}{part}_proj.weight"] = gv
        else:
            t[f"{name}.{n}"] = p
            t[f"{name}.grad_{n}"] = g


def group_transformer():
    """S10/S11: MultiheadAttention and a Pre-LN TransformerEncoderLayer, with torch's
    weights loaded into Volta's `MultiHeadAttention` and `TransformerBlock`.

    Neither torch module adds a residual inside attention. The block is
    `x + Attn(LN1(x))`, then `+ FFN(LN2(.))`, with a GELU (erf) FFN and no dropout.
    """
    torch.manual_seed(SEED)
    t = {}
    E, H, B, L = 8, 2, 2, 3
    causal = torch.triu(torch.full((L, L), -INF), diagonal=1)

    for is_causal in (False, True):
        name = "mha_causal" if is_causal else "mha"
        mha = torch.nn.MultiheadAttention(E, H, dropout=0.0, bias=True, batch_first=True)
        with torch.no_grad():
            mha.in_proj_bias.zero_()
        x = torch.randn(B, L, E).requires_grad_(True)
        mask = causal if is_causal else None
        out, _ = mha(x, x, x, attn_mask=mask, need_weights=False, is_causal=is_causal)
        module_case(t, name, mha, x, out, {"in_proj_weight"})

    for is_causal in (False, True):
        name = "block_causal" if is_causal else "block"
        block = torch.nn.TransformerEncoderLayer(
            E, H, dim_feedforward=2 * E, dropout=0.0, activation="gelu",
            layer_norm_eps=1e-5, batch_first=True, norm_first=True, bias=True,
        )
        block.train()  # keeps torch off its fused inference fast path
        with torch.no_grad():
            block.self_attn.in_proj_bias.zero_()
        x = torch.randn(B, L, E).requires_grad_(True)
        mask = causal if is_causal else None
        out = block(x, src_mask=mask, is_causal=is_causal)
        module_case(t, name, block, x, out, {"self_attn.in_proj_weight"})

    save("transformer", t)


def group_f64():
    """Forward and backward in float64 for ops, every layer kind, and Adam (Volta P3b).

    Inputs and weights are f32 random values widened to f64, so the f32 state dicts Volta
    loads carry them exactly; everything after that is computed in f64. Adam's
    hyperparameters are powers of two for the same reason (Volta's are f32). Stored in
    native f64.
    """
    torch.manual_seed(SEED)
    F = torch.nn.functional
    t = {}

    def rl(*shape):
        return torch.randn(*shape).double().requires_grad_(True)

    def case64(name, out, inputs):
        w = torch.randn_like(out)
        names = list(inputs)
        grads = torch.autograd.grad((out * w).sum(), [inputs[n] for n in names])
        t[f"{name}.out"] = out
        t[f"{name}.w"] = w
        for n, g in zip(names, grads):
            t[f"{name}.grad_{n}"] = g

    x = rl(3, 4)
    t["ops.x"] = x
    for op in ["exp", "tanh", "sigmoid", "erf"]:
        case64(f"ops.{op}", getattr(torch, op)(x), {"x": x})
    case64("ops.softmax", torch.softmax(x, 1), {"x": x})
    case64("ops.log_softmax", torch.log_softmax(x, 0), {"x": x})
    a, b = rl(2, 1, 3, 4), rl(1, 3, 4, 2)
    t.update({"matmul.a": a, "matmul.b": b})
    case64("matmul", a @ b, {"a": a, "b": b})

    x, w, bias = rl(4, 2), rl(3, 2), rl(3)
    t.update({"linear.x": x, "linear.weight": w, "linear.bias": bias})
    case64("linear", F.linear(x, w, bias), {"x": x, "weight": w, "bias": bias})

    x, w, bias = rl(2, 2, 5, 5), rl(3, 2, 3, 3), rl(3)
    t.update({"conv.x": x, "conv.weight": w, "conv.bias": bias})
    case64("conv", F.conv2d(x, w, bias, stride=2, padding=1), {"x": x, "weight": w, "bias": bias})

    x, w, bias = rl(2, 2, 3, 3), rl(2, 3, 3, 3), rl(3)
    t.update({"convt.x": x, "convt.weight": w, "convt.bias": bias})
    case64("convt", F.conv_transpose2d(x, w, bias, stride=2, padding=1), {"x": x, "weight": w, "bias": bias})

    x = rl(2, 2, 4, 4)
    t["maxpool.x"] = x
    case64("maxpool", F.max_pool2d(x, 2, 2), {"x": x})

    x, g, bias = rl(3, 2, 2, 2), rl(2), rl(2)
    t.update({"batchnorm.x": x, "batchnorm.gamma": g, "batchnorm.beta": bias})
    out = F.batch_norm(x, None, None, g, bias, training=True, eps=1e-5)
    case64("batchnorm", out, {"x": x, "gamma": g, "beta": bias})

    x, g, bias = rl(3, 4), rl(4), rl(4)
    t.update({"layernorm.x": x, "layernorm.gamma": g, "layernorm.beta": bias})
    case64("layernorm", F.layer_norm(x, (4,), g, bias, 1e-5), {"x": x, "gamma": g, "beta": bias})

    w = rl(5, 3)
    t["embedding.weight"] = w
    case64("embedding", F.embedding(torch.tensor([1, 3, 1]), w), {"weight": w})

    p = torch.randn(2, 3).double().requires_grad_(True)
    t["adam.init"] = p.detach().clone()
    opt = torch.optim.Adam([p], lr=2**-7, betas=(0.875, 1 - 2**-10), eps=2**-27, weight_decay=2**-7)
    for k in range(3):
        g = torch.randn(2, 3).double()
        p.grad = g.clone()
        t[f"adam.g{k}"] = g
        opt.step()
        t[f"adam.after{k}"] = p.detach().clone()

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"info": f"torch={torch.__version__};seed={SEED};group=f64"}
    save_file({k: v.detach().contiguous() for k, v in t.items()}, OUT / "f64.safetensors", metadata=meta)
    print(f"wrote f64: {len(t)} tensors")


def group_half():
    """Forward and backward in float16 and bfloat16 for every op family (Volta P4).

    Each case runs twice, keys prefixed `f16.` and `bf16.`. Inputs are random values
    rounded to the dtype, and torch computes natively in it (CPU: each op decodes to f32,
    computes, and rounds its output). Layer weights are half values, which Volta's f32
    state dicts carry exactly. Stored in the native half dtype.
    """
    F = torch.nn.functional
    t = {}
    for dname, dt in [("f16", torch.float16), ("bf16", torch.bfloat16)]:
        torch.manual_seed(SEED)

        def rl(*shape):
            return torch.randn(*shape).to(dt).requires_grad_(True)

        def case(name, out, inputs):
            w = torch.randn(out.shape).to(dt)
            names = list(inputs)
            grads = torch.autograd.grad((out * w).sum(), [inputs[n] for n in names])
            t[f"{dname}.{name}.out"] = out
            t[f"{dname}.{name}.w"] = w
            for n, g in zip(names, grads):
                t[f"{dname}.{name}.grad_{n}"] = g

        def leaf(name, x):
            t[f"{dname}.{name}"] = x
            return x

        x = leaf("unary.x", rl(3, 4))
        xp = leaf("unary.xp", (torch.rand(3, 4) * 2 + 0.25).to(dt).requires_grad_(True))
        for op in ["neg", "exp", "exp2", "sin", "cos", "tanh", "sigmoid", "relu", "erf"]:
            case(f"unary.{op}", getattr(torch, op)(x), {"x": x})
        for op, f in [("recip", torch.reciprocal), ("sqrt", torch.sqrt), ("log", torch.log), ("log2", torch.log2)]:
            case(f"unary.{op}", f(xp), {"x": xp})

        a, b = leaf("binary.a", rl(3, 4)), leaf("binary.b", rl(4))
        bp = leaf("binary.bp", (torch.rand(4) + 0.5).to(dt).requires_grad_(True))
        for op, f in [("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("max", torch.maximum)]:
            case(f"binary.{op}", f(a, b), {"a": a, "b": b})
        case("binary.div", a / bp, {"a": a, "b": bp})

        x = leaf("reduce.x", rl(3, 5))
        case("reduce.sum", x.sum(), {"x": x})
        case("reduce.mean", x.mean(), {"x": x})
        case("reduce.max", x.max(), {"x": x})
        case("reduce.sum_dim", x.sum(1), {"x": x})
        case("reduce.mean_dim", x.mean(0, keepdim=True), {"x": x})
        case("reduce.max_dim", x.amax(1), {"x": x})
        case("reduce.softmax", torch.softmax(x, 1), {"x": x})
        case("reduce.log_softmax", torch.log_softmax(x, 0), {"x": x})

        x = leaf("move.x", rl(2, 3, 4))
        case("move.permute", x.permute(2, 0, 1), {"x": x})
        case("move.reshape", x.reshape(6, 4), {"x": x})
        case("move.shrink", x[:, 1:3, 0:2], {"x": x})
        case("move.expand", x[:, :1, :].expand(2, 5, 4), {"x": x})
        m = leaf("move.m", rl(3, 5))
        case("move.transpose", m.t(), {"x": m})

        a, b = leaf("matmul.a", rl(2, 1, 3, 4)), leaf("matmul.b", rl(1, 3, 4, 2))
        case("matmul", a @ b, {"a": a, "b": b})

        x, y, z = leaf("ternary.x", rl(3, 4)), leaf("ternary.y", rl(3, 4)), leaf("ternary.z", rl(3, 4))
        case("ternary.mulacc", torch.addcmul(z, x, y), {"x": x, "y": y, "z": z})
        cond = leaf("ternary.cond", (torch.rand(3, 4) > 0.5).to(dt))
        case("ternary.where", torch.where(cond != 0, y, z), {"y": y, "z": z})

        p, q = leaf("loss.pred", rl(4, 3)), leaf("loss.target", rl(4, 3))
        case("loss.mse", F.mse_loss(p, q), {"pred": p})
        onehot = leaf("loss.onehot", F.one_hot(torch.tensor([0, 2, 1, 2]), 3).to(dt))
        case("loss.cross_entropy", -(onehot * torch.log_softmax(p, 1)).sum(1).mean(), {"pred": p})

        x, w, bias = leaf("linear.x", rl(4, 2)), leaf("linear.weight", rl(3, 2)), leaf("linear.bias", rl(3))
        case("linear", F.linear(x, w, bias), {"x": x, "weight": w, "bias": bias})

        x, w, bias = leaf("conv.x", rl(2, 2, 5, 5)), leaf("conv.weight", rl(3, 2, 3, 3)), leaf("conv.bias", rl(3))
        case("conv", F.conv2d(x, w, bias, stride=2, padding=1), {"x": x, "weight": w, "bias": bias})

        x, w, bias = leaf("convt.x", rl(2, 2, 3, 3)), leaf("convt.weight", rl(2, 3, 3, 3)), leaf("convt.bias", rl(3))
        case("convt", F.conv_transpose2d(x, w, bias, stride=2, padding=1), {"x": x, "weight": w, "bias": bias})

        x = leaf("maxpool.x", rl(2, 2, 4, 4))
        case("maxpool", F.max_pool2d(x, 2, 2), {"x": x})

        x, g, bias = leaf("batchnorm.x", rl(3, 2, 2, 2)), leaf("batchnorm.gamma", rl(2)), leaf("batchnorm.beta", rl(2))
        case("batchnorm", F.batch_norm(x, None, None, g, bias, training=True, eps=1e-5), {"x": x, "gamma": g, "beta": bias})

        x, g, bias = leaf("layernorm.x", rl(3, 4)), leaf("layernorm.gamma", rl(4)), leaf("layernorm.beta", rl(4))
        case("layernorm", F.layer_norm(x, (4,), g, bias, 1e-5), {"x": x, "gamma": g, "beta": bias})

        w = leaf("embedding.weight", rl(5, 3))
        case("embedding", F.embedding(torch.tensor([1, 3, 1]), w), {"weight": w})

        for oname, make in [
            ("sgd", lambda ps: torch.optim.SGD(ps, lr=2**-4, momentum=0.875)),
            ("adam", lambda ps: torch.optim.Adam(ps, lr=2**-7, betas=(0.875, 1 - 2**-10), eps=2**-14, weight_decay=2**-7)),
        ]:
            p = torch.randn(2, 3).to(dt).requires_grad_(True)
            t[f"{dname}.{oname}.init"] = p.detach().clone()
            opt = make([p])
            for k in range(3):
                g = torch.randn(2, 3).to(dt)
                p.grad = g.clone()
                t[f"{dname}.{oname}.g{k}"] = g
                opt.step()
                t[f"{dname}.{oname}.after{k}"] = p.detach().clone()

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"info": f"torch={torch.__version__};seed={SEED};group=half"}
    save_file({k: v.detach().clone().contiguous() for k, v in t.items()}, OUT / "half.safetensors", metadata=meta)
    print(f"wrote half: {len(t)} tensors")


def group_intbool():
    """I64 and Bool tensors as data (Volta P5): comparisons, `where` with a bool
    condition, cross-entropy and NLL with class-index targets, `Embedding` with 2-D
    indices, and movement ops on i64 and bool. Stored in native dtypes.
    """
    torch.manual_seed(SEED)
    F = torch.nn.functional
    t = {}

    def case(name, out, inputs):
        w = torch.randn(out.shape)
        names = list(inputs)
        grads = torch.autograd.grad((out * w).sum(), [inputs[n] for n in names])
        t[f"{name}.out"] = out
        t[f"{name}.w"] = w
        for n, g in zip(names, grads):
            t[f"{name}.grad_{n}"] = g

    # Floats with ties and a NaN; b broadcasts along the rows.
    a = torch.tensor([[0.5, -1.0, 2.0, NAN], [1.5, 0.5, -1.0, 3.0], [2.0, 2.0, 0.0, -0.5]])
    b = torch.tensor([0.5, 0.0, 2.0, 1.0])
    ia = torch.tensor([[3, -7, 1 << 40], [0, 5, -(1 << 40)]], dtype=torch.int64)
    ib = torch.tensor([0, 5, 1 << 40], dtype=torch.int64)
    t.update({"cmp.a": a, "cmp.b": b, "cmp.ia": ia, "cmp.ib": ib})
    for op in ["lt", "le", "gt", "ge", "eq", "ne"]:
        t[f"cmp.{op}"] = getattr(torch, op)(a, b)
        t[f"cmp.int_{op}"] = getattr(torch, op)(ia, ib)

    cond = torch.tensor([[True], [False], [True]])
    y = torch.randn(3, 4).requires_grad_(True)
    z = torch.randn(4).requires_grad_(True)
    t.update({"where.cond": cond, "where.y": y, "where.z": z})
    case("where", torch.where(cond, y, z), {"y": y, "z": z})
    iy = torch.arange(12, dtype=torch.int64).reshape(3, 4) << 33
    t.update({"where.iy": iy, "where.int_out": torch.where(cond, iy, torch.tensor(-1, dtype=torch.int64))})

    logits = torch.randn(5, 4).requires_grad_(True)
    target = torch.tensor([3, 0, 1, 3, 2], dtype=torch.int64)
    t.update({"ce.logits": logits, "ce.target": target})
    out = F.cross_entropy(logits, target)
    t["ce.out"] = out
    t["ce.grad_logits"] = torch.autograd.grad(out, logits)[0]
    lp = torch.log_softmax(torch.randn(5, 4), 1).requires_grad_(True)
    t["nll.logp"] = lp
    out = F.nll_loss(lp, target)
    t["nll.out"] = out
    t["nll.grad_logp"] = torch.autograd.grad(out, lp)[0]

    weight = torch.randn(6, 3).requires_grad_(True)
    idx = torch.tensor([[1, 5, 1], [0, 1, 4]], dtype=torch.int64)
    t.update({"embedding.weight": weight, "embedding.idx": idx})
    case("embedding", F.embedding(idx, weight), {"weight": weight})

    x = (torch.arange(24, dtype=torch.int64).reshape(2, 3, 4) - 7) << 41
    m = torch.rand(2, 3, 4) > 0.5
    t.update({"move.x": x, "move.m": m})
    for name, v in [("x", x), ("m", m)]:
        t[f"move.{name}.permute"] = v.permute(2, 0, 1)
        t[f"move.{name}.shrink"] = v[:, 1:3, 0:2]
        t[f"move.{name}.expand"] = v[:, :1, :].expand(2, 5, 4)
        t[f"move.{name}.transpose"] = v[0].t()
        t[f"move.{name}.pad"] = F.pad(v[0].to(torch.int64), (1, 2)).to(v.dtype)

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"info": f"torch={torch.__version__};seed={SEED};group=intbool"}
    save_file({k: v.detach().clone().contiguous() for k, v in t.items()}, OUT / "intbool.safetensors", metadata=meta)
    print(f"wrote intbool: {len(t)} tensors")


def group_promotion():
    """Implicit promotion (Volta P6): for every ordered pair of Volta's eight dtypes,
    torch's result for add, mul, div, lt and where(cond, a, b), stored in torch's result
    dtype (so the file records the dtype too), and for float pairs the gradients of
    sum(a * b * w) in each operand's own dtype. The operands are small integers cast from
    f32 (`a`, `b`), exact in every dtype; Volta casts them the same way.
    """
    torch.manual_seed(SEED)
    names = ["f16", "bf16", "f32", "f64", "i32", "i64", "u8", "bool"]
    a = torch.tensor([0.0, 1.0, 2.0, 3.0])
    b = torch.tensor([3.0, 2.0, 1.0, 1.0])
    cond = torch.tensor([True, False, True, False])
    w = torch.tensor([0.5, -1.0, 2.0, 0.25])
    t = {"a": a, "b": b, "cond": cond, "w": w}
    for na, da in zip(names, VOLTA_DTYPES):
        for nb, db in zip(names, VOLTA_DTYPES):
            x, y = a.to(da), b.to(db)
            key = f"{na}.{nb}"
            t[f"add.{key}"] = x + y
            t[f"mul.{key}"] = x * y
            t[f"div.{key}"] = x / y
            t[f"lt.{key}"] = torch.lt(x, y)
            t[f"where.{key}"] = torch.where(cond, x, y)
            if da.is_floating_point and db.is_floating_point:
                x, y = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
                gx, gy = torch.autograd.grad((x * y * w.to(torch.promote_types(da, db))).sum(), [x, y])
                t[f"grad_a.{key}"], t[f"grad_b.{key}"] = gx, gy

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"info": f"torch={torch.__version__};seed={SEED};group=promotion"}
    save_file({k: v.detach().clone().contiguous() for k, v in t.items()}, OUT / "promotion.safetensors", metadata=meta)
    print(f"wrote promotion: {len(t)} tensors")


def group_autocast():
    """CPU autocast in bf16 and f16 (Volta P7): f32 parameters and inputs, forward under
    `torch.autocast("cpu", dtype=..)`, backward outside it. Cases: an MLP (Linear, ReLU,
    Linear) with class-index cross-entropy (loss and every parameter's f32 gradient),
    Conv2d and ConvTranspose2d (output in the low dtype and the f32 gradients of input,
    weight and bias under a weighted sum), broadcast batched matmul, and LayerNorm after
    a Linear and BatchNorm2d after a Conv2d (half input, f32 parameters). Keys prefixed
    `bf16.` / `f16.`; stored in native dtypes.
    """
    F = torch.nn.functional
    t = {}
    for dname, dt in [("bf16", torch.bfloat16), ("f16", torch.float16)]:
        torch.manual_seed(SEED)

        def rl(*shape):
            return torch.randn(*shape).requires_grad_(True)

        def case(name, fwd, inputs):
            with torch.autocast("cpu", dtype=dt):
                out = fwd()
            w = torch.randn(out.shape)
            names = list(inputs)
            grads = torch.autograd.grad((out * w).sum(), [inputs[n] for n in names])
            t[f"{dname}.{name}.out"] = out
            t[f"{dname}.{name}.w"] = w
            for n, g in zip(names, grads):
                t[f"{dname}.{name}.grad_{n}"] = g

        x, w1, b1, w2, b2 = rl(5, 8), rl(16, 8), rl(16), rl(3, 16), rl(3)
        y = torch.tensor([0, 2, 1, 1, 0], dtype=torch.int64)
        t.update({f"{dname}.mlp.{k}": v for k, v in dict(x=x, w1=w1, b1=b1, w2=w2, b2=b2, y=y).items()})
        with torch.autocast("cpu", dtype=dt):
            loss = F.cross_entropy(F.linear(torch.relu(F.linear(x, w1, b1)), w2, b2), y)
        t[f"{dname}.mlp.loss"] = loss
        for k, g in zip(["x", "w1", "b1", "w2", "b2"], torch.autograd.grad(loss, [x, w1, b1, w2, b2])):
            t[f"{dname}.mlp.grad_{k}"] = g

        x, w, b = rl(2, 2, 5, 5), rl(3, 2, 3, 3), rl(3)
        t.update({f"{dname}.conv.x": x, f"{dname}.conv.weight": w, f"{dname}.conv.bias": b})
        case("conv", lambda: F.conv2d(x, w, b, stride=2, padding=1), {"x": x, "weight": w, "bias": b})

        x, w, b = rl(2, 2, 3, 3), rl(2, 3, 3, 3), rl(3)
        t.update({f"{dname}.convt.x": x, f"{dname}.convt.weight": w, f"{dname}.convt.bias": b})
        case("convt", lambda: F.conv_transpose2d(x, w, b, stride=2, padding=1), {"x": x, "weight": w, "bias": b})

        a, bm = rl(2, 1, 3, 4), rl(1, 3, 4, 2)
        t.update({f"{dname}.matmul.a": a, f"{dname}.matmul.b": bm})
        case("matmul", lambda: a @ bm, {"a": a, "b": bm})

        # Norms after a low-precision op: torch normalizes the half input with f32
        # parameters in f32 and returns the input's dtype (Volta P8 audit A5).
        x, w, b, g, be = rl(4, 6), rl(5, 6), rl(5), rl(5), rl(5)
        t.update({f"{dname}.ln.{k}": v for k, v in dict(x=x, weight=w, bias=b, gamma=g, beta=be).items()})
        case("ln", lambda: F.layer_norm(F.linear(x, w, b), (5,), g, be, 1e-5), {"x": x, "weight": w, "bias": b, "gamma": g, "beta": be})
        x, w, g, be = rl(3, 2, 4, 4), rl(2, 2, 3, 3), rl(2), rl(2)
        t.update({f"{dname}.bn.{k}": v for k, v in dict(x=x, weight=w, gamma=g, beta=be).items()})
        case("bn", lambda: F.batch_norm(F.conv2d(x, w, None, padding=1), None, None, g, be, training=True, eps=1e-5), {"x": x, "weight": w, "gamma": g, "beta": be})

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"info": f"torch={torch.__version__};seed={SEED};group=autocast"}
    save_file({k: v.detach().clone().contiguous() for k, v in t.items()}, OUT / "autocast.safetensors", metadata=meta)
    print(f"wrote autocast: {len(t)} tensors")


# Volta's `DType` discriminants, in order: F16=0, BF16=1, F32=2, F64=3, I32=4, I64=5, U8=6, Bool=7.
VOLTA_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8, torch.bool]


def group_dtypes():
    """`promotion[i, j]` is the Volta code of `torch.result_type` for two 1-element tensors.

    Dimensioned tensors on purpose: zero-dim tensors and Python scalars promote by category
    only, which is a separate rule (ADDENDUM §11, P6).
    """
    codes = [
        [VOLTA_DTYPES.index(torch.result_type(torch.zeros(1, dtype=a), torch.zeros(1, dtype=b))) for b in VOLTA_DTYPES]
        for a in VOLTA_DTYPES
    ]
    save("dtypes", {"promotion": torch.tensor(codes, dtype=torch.float32)})


NAMES = ["f16", "bf16", "f32", "f64", "i32", "i64", "u8", "bool"]

# Inputs per source dtype, as Python values; torch rounds them into the source dtype.
FLOAT_FULL = [
    0.0, -0.0, 1.0, -1.5, 2.5, 0.1, 1 / 3, 65504.0, 65520.0, 1e-8, 6e-8, 3.0e38, 1e300,
    float("inf"), float("-inf"), float("nan"),
    # Halfway cases: 1 + 2^-11 is the f16 tie above 1; + 2^-30 is just above it, which
    # rounds up directly but ties-to-even down if rounded through f32 first.
    1 + 2**-11, 1 + 2**-11 + 2**-30, 1 + 2**-8 + 2**-30,
]
# Float values every integer dtype can hold after truncation toward zero.
FLOAT_SIGNED = [0.0, -0.0, 0.9, -0.9, 1.5, -1.5, 2.7, -2.7, 100.9, -100.9, 127.5]
# 250.5, not 255.9: bf16 rounds 255.9 to 256, outside u8.
FLOAT_UNSIGNED = [0.0, 0.9, 1.5, 2.7, 100.9, 200.5, 250.5]
INTS = {
    torch.int32: [0, 1, -1, 127, 128, 255, 256, -129, 2**24 + 1, 2**31 - 1, -(2**31)],
    torch.int64: [0, 1, -1, 255, 256, -129, 2**24 + 1, 2**31, 2**53 + 1, 2**63 - 1, -(2**63)],
    torch.uint8: [0, 1, 127, 128, 255],
    torch.bool: [True, False],
}


def group_casts():
    """`x.to(dst)` for every pair of Volta dtypes, stored in their native dtypes.

    `cast.<src>.<set>` is the input and `cast.<src>.<set>.<dst>` torch's result. Float
    sources have three sets: `full` (cast to floats and bool), `signed` (to i32/i64), and
    `unsigned` (to u8); out-of-range float-to-int is undefined in torch, so it is left out.
    """
    out = {}
    for src, sname in zip(VOLTA_DTYPES, NAMES):
        if src.is_floating_point:
            sets = {
                "full": (FLOAT_FULL, [d for d in VOLTA_DTYPES if d.is_floating_point or d == torch.bool]),
                "signed": (FLOAT_SIGNED, [torch.int32, torch.int64]),
                "unsigned": (FLOAT_UNSIGNED, [torch.uint8]),
            }
        else:
            sets = {"all": (INTS[src], VOLTA_DTYPES)}
        for set_name, (values, dsts) in sets.items():
            x = torch.tensor(values, dtype=src)
            out[f"cast.{sname}.{set_name}"] = x
            for dst in dsts:
                out[f"cast.{sname}.{set_name}.{NAMES[VOLTA_DTYPES.index(dst)]}"] = x.to(dst).clone()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"info": f"torch={torch.__version__};group=casts"}
    save_file({k: v.contiguous() for k, v in out.items()}, OUT / "casts.safetensors", metadata=meta)
    print(f"wrote casts: {len(out)} tensors")


GROUPS = {"ops": group_ops, "reductions": group_reductions, "optim": group_optim, "layers": group_layers, "transformer": group_transformer, "dtypes": group_dtypes, "casts": group_casts, "f64": group_f64, "half": group_half, "intbool": group_intbool, "promotion": group_promotion, "autocast": group_autocast}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(GROUPS)
    for name in wanted:
        GROUPS[name]()
