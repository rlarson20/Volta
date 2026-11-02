# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "safetensors", "numpy"]
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
    a, b = leaf([NAN, 1.0]), leaf([1.0, NAN])
    t.update({"max_elem_nan.a": a, "max_elem_nan.b": b, "max_elem_nan.out": torch.maximum(a, b)})

    # S4: full max() with a tie splits the gradient evenly.
    x = leaf([2.0, 5.0, 5.0, 1.0])
    out = x.max()
    (gx,) = grad_of(out, x)
    t.update({"max_all_tie.x": x, "max_all_tie.out": out.reshape(1), "max_all_tie.grad": gx})

    # S4: full max() with NaN returns NaN.
    x = leaf([1.0, NAN, 3.0])
    t.update({"max_all_nan.x": x, "max_all_nan.out": x.max().reshape(1)})

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

    # S7: BCE at the clamp (log clamped to >= -100).
    p, y = leaf([0.0, 1.0, 0.0, 1.0, 0.3]), torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    out = torch.nn.functional.binary_cross_entropy(p, y, reduction="mean")
    (gp,) = grad_of(out, p)
    t.update({"bce.pred": p, "bce.target": y, "bce.out": out.reshape(1), "bce.grad": gp})

    # S8: cross entropy where a non-target class has logit -inf.
    z = leaf([[1.0, -INF, 2.0], [0.5, 0.2, -INF]])
    cls = torch.tensor([0, 1])
    out = torch.nn.functional.cross_entropy(z, cls)
    t.update({"ce_neginf.logits": z, "ce_neginf.classes": cls.float(), "ce_neginf.out": out.reshape(1)})

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


def case(t, name, out, inputs, seed_w=True):
    """Record `out`, a random weighting `w`, and the gradient of sum(out * w) per input.

    `inputs` maps a short name to a leaf tensor already stored by the caller.
    """
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


GROUPS = {"ops": group_ops, "reductions": group_reductions, "optim": group_optim, "layers": group_layers}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(GROUPS)
    for name in wanted:
        GROUPS[name]()
