"""
List of todos:

1. have the hellaswag run as validation option. keep the rest same.
"""
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask
import torch.nn.functional as F
from torch import Tensor, nn
import torch
import os
import sys
import time
import glob
from dataclasses import dataclass
from pathlib import Path
from itertools import cycle
from hashlib import sha256
from typing import Literal


# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz (Single-GPU Version)

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix using Newton-Schulz iteration.

    Warnings:
    - This optimizer should only be used for parameters that are matrix-like (2D or can be reshaped to 2D).
    - Exclude embeddings, final layers, biases, and normalization parameters.

    Arguments:
        params: Iterable of parameters to optimize. Should ideally only contain 2D+ parameters.
        lr: The learning rate used by the internal SGD.
        momentum: The momentum factor.
        nesterov: Whether to use Nesterov-style momentum. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not ns_steps >= 1:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")

        defaults = dict(lr=lr, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon optimizer requires all parameters to be 2D+, but found parameter "
                        f"with shape {p.shape} (ndim={p.ndim}) in group {group.get('name', 'default')}. "
                    )
        pass

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # State initialization (momentum buffer)
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)  # Match grad dtype/device

                buf = state['momentum_buffer']

                # Update momentum buffer: buf = momentum * buf + (1 - momentum) * g
                # Using lerp as in the original code for consistency:
                buf.lerp_(g, 1 - momentum)

                # Calculate the update direction d_p based on momentum type
                if nesterov:
                    # Original code used: g.lerp_(buf, momentum) -> (1-momentum)*g + momentum*buf
                    d_p = g.lerp(buf, momentum)
                else:
                    # Original code used: buf
                    d_p = buf

                # --- Orthogonalization Step ---
                d_p_orth = zeropower_via_newtonschulz5(d_p, steps=ns_steps)
                # --- End Orthogonalization ---

                scaling_factor = max(1.0, p.size(-2) / p.size(-1))**0.5

                # Apply the orthogonalized update to the parameter
                # p = p - lr * scaling_factor * d_p_orth
                p.add_(d_p_orth, alpha=-lr * scaling_factor)
        pass

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1,
                                                    steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat(
            [angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None,
                            :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std  # improved init scale by @YouJiacheng
        self.qkv_w = nn.Parameter(torch.empty(
            3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(
            B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * \
                ve.view_as(v)  # @KoszarskyB & @Grad62304977
        else:  # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(
            1, 2), block_mask=block_mask, scale=0.12).transpose(1, 2)
        # re-assemble all head outputs side by side
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(
            dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model


def next_multiple_of_n(v: float | int, *, n: int):
    return int((-(-v // n)) * n)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList(
            [nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(
            model_dim, next_multiple_of_n(vocab_size, n=128))

        self.lm_head.weight.detach().zero_()  # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def create_blockmasks(self, input_seq: Tensor):
        EOD = 50256  # End of document, a delimiter
        docs = (input_seq == EOD).cumsum(0)

        s_window, long_window = 128, 1024

        def long_mask_f(b, h, q_idx, kv_idx):
            causal_m = q_idx >= kv_idx
            document_m = docs[q_idx] == docs[kv_idx]
            window_m = q_idx - kv_idx < long_window
            return causal_m & document_m & window_m

        def short_mask_f(b, h, q_idx, kv_idx):
            causal_m = q_idx >= kv_idx
            document_m = docs[q_idx] == docs[kv_idx]
            window_m = q_idx - kv_idx < s_window
            return causal_m & document_m & window_m

        s_len = len(input_seq)
        l_mask = create_block_mask(long_mask_f, None, None, s_len, s_len)
        s_mask = create_block_mask(short_mask_f, None, None, s_len, s_len)
        return l_mask, s_mask

    def forward(self, input_seq: Tensor,
                target_seq: Tensor,
                reduction: Literal["sum", "mean", "none"] = "mean"
                ):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * \
            (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm,
                       short_bm, short_bm, long_bm, short_bm, long_bm, short_bm, short_bm]
        assert len(block_masks) == len(self.blocks)

        # use of norm here by @Grad62304977
        x = x0 = norm(self.embed(input_seq)[None])

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        if self.training:
            reduction = "sum"

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target_seq, reduction=reduction)
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256,
                             dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        # avoid pin_memory copy by @YouJiacheng
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        # avoid bytes->array copy by @YouJiacheng
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def data_generator(filename_pattern: str, batch_size: int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    file_iter = cycle(iter(files))

    tokens, pos = _load_data_shard(next(file_iter)), 0

    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0

        buf = tokens[pos:][:batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32,
                             non_blocking=True)  # no sync on host side;
        targets = buf[1:].to(
            device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main


@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    # input .bin to eval validation loss on
    val_files = "data/fineweb10B/fineweb_val_*.bin"
    # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    val_tokens = 10485760
    train_seq_len = 48 * 1024  # FlexAttention sequence length
    val_seq_len = 4 * 64 * 1024  # FlexAttention sequence length for validation
    # optimization
    num_iterations = 3000  # number of iterations to run
    cooldown_frac = 0.4  # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = True


args = Hyperparameters()

assert torch.cuda.is_available(), "We need cuda"
device = torch.device("cuda", 0)


class CustomLogger:
    def __init__(self, dry=False):
        """
        dry: bool = False, don't create a log file
        """
        self.dry = dry
        os.makedirs("logs", exist_ok=True)
        code = self._get_code()
        self.run_id = self._get_id(code)
        self.filename = f"./logs/{self.run_id}.txt"
        if not dry:
            self._build_starter()

    def _get_code(self):
        with open(sys.argv[0]) as f:
            code = f.read()
        return code

    def get_id(self):
        return self.run_id

    def _get_id(self, code):
        sha_hash = sha256(code.encode("utf-8")).hexdigest()[:8]
        timestamp = int(time.time())
        return f"{timestamp}_{sha_hash}"

    def _build_starter(self):
        with open(sys.argv[0]) as f:
            code = f.read()  # read the code of this file ASAP, for logging
        l = self.log
        # print date and time
        l(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        l(f"Running Python {sys.version}")
        l(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
        self.divider()
        l(code)
        self.divider()
        l(self._nvidia_smi())
        self.divider()

    def divider(self):
        self.log("="*100)

    def log(self, x):
        print(x)
        if self.dry:
            return
        with open(self.filename, mode="a") as f:
            f.write(x+"\n")

    def _nvidia_smi(self):
        import subprocess  # avoid top level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


logger = CustomLogger()
run_id = logger.get_id()
print0 = logger.log

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda().bfloat16()


def create_optimizer(model: nn.Module):
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters(
    ) if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # init the optimizer(s)
    adam_params = [dict(params=head_params, lr=0.22), dict(
        params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(
        adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05,
                      momentum=0.95)
    optimizers = [optimizer1, optimizer2]

    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return optimizers


optimizers = create_optimizer(model)


def get_lr(step: int):
    x = step / args.num_iterations  # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1


model: nn.Module = torch.compile(model, dynamic=False)


@torch.no_grad()
def val_run() -> float:
    """
    Run a validation pass on the model.
    """
    model.eval()
    val_loader = data_generator(args.val_files, args.val_seq_len)
    val_loss = 0
    steps = args.val_tokens // args.val_seq_len
    for _ in range(steps):
        inputs, targets = next(val_loader)
        val_loss += model(inputs, targets)
    val_loss /= (steps)
    model.train()
    del val_loader
    return val_loss


@torch.no_grad()
def hellaswag_run() -> float:
    model.eval()

    model.train()

########################################
#        Training and validation       #
########################################


train_loader = data_generator(args.train_files, args.train_seq_len)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)

        val_loss = val_run()

        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms")
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if args.save_checkpoint:
            log = dict(step=step, model=model.state_dict(),
                       optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    # to make it identical to the multi GPU run
    microbatch_count = 8
    for _ in range(microbatch_count):
        inputs, targets = next(train_loader)
        loss = model(inputs, targets)
        loss = loss / microbatch_count
        loss.backward()

    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)

    # for the muon params
    for group in optimizers[1].param_groups:
        frac = min(step / 300, 1)  # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad()
    # logging
    approx_training_time_ms = training_time_ms + \
        1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms")

    pass
