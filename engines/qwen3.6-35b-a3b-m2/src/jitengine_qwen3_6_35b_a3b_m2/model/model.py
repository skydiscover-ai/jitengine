"""Qwen3.6-MoE (qwen3_5_moe) forward: Gated DeltaNet and gated attention blocks, each with a MoE.

Matches mlx_lm op for op and dtype for dtype, so greedy output is token-exact with mlx_lm.
`forward` runs one row; `decode_step` runs B >= 2 decode rows, reading each shared
weight once for the batch while per-request state stays per row.
"""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.kvcache.state import RequestState
from jitengine_qwen3_6_35b_a3b_m2.linear import (
    FusedLinear,
    QEmbedding,
    QLinear,
    calibrate_row_limits,
)
from jitengine_qwen3_6_35b_a3b_m2.model.attention import Qwen3_5MoeAttention
from jitengine_qwen3_6_35b_a3b_m2.model.config import Qwen3_5MoeConfig
from jitengine_qwen3_6_35b_a3b_m2.model.gdn import Qwen3_5MoeGatedDeltaNet
from jitengine_qwen3_6_35b_a3b_m2.model.kernel_choice import KernelChoice, Strategies
from jitengine_qwen3_6_35b_a3b_m2.model.moe import Qwen3_5MoeSparseMoeBlock
from jitengine_qwen3_6_35b_a3b_m2.model.weight import FUSIONS


class Qwen3_5MoeDecoderLayer:
    def __init__(
        self, cfg: Qwen3_5MoeConfig, w: dict, i: int, pool=None, fuse: frozenset = FUSIONS
    ):
        p = f"model.layers.{i}."
        self.is_attention = cfg.is_attention(i)
        self.block = (
            Qwen3_5MoeAttention(cfg, w, p + "self_attn.", fuse)
            if self.is_attention
            else Qwen3_5MoeGatedDeltaNet(cfg, w, p + "linear_attn.", fuse)
        )
        self.moe = Qwen3_5MoeSparseMoeBlock(cfg, w, p + "mlp.", pool=pool, layer_idx=i, fuse=fuse)
        self.paths = KernelChoice()
        self.ln1 = w[p + "input_layernorm.weight"]
        self.ln2 = w[p + "post_attention_layernorm.weight"]
        self.eps = cfg.rms_norm_eps

    def __call__(self, x: mx.array, slot, offset: int) -> mx.array:
        if self.paths.fused_norm and x.size == x.shape[-1]:
            # one launch: the norm (MLX's arithmetic) + the pre-scale the prescaled GEMVs after it need
            from jitengine_qwen3_6_35b_a3b_m2.kernels.rms_norm_prescale import rms_norm_pre

            xn, pre = rms_norm_pre(x, self.ln1, self.eps)
            h = x + self.block(xn, slot, offset, pre=pre)
            hn, pre2 = rms_norm_pre(h, self.ln2, self.eps)
            return self.moe.with_residual(hn, h, pre=pre2)
        h = x + self.block(mx.fast.rms_norm(x, self.ln1, self.eps), slot, offset)
        return self.moe.with_residual(mx.fast.rms_norm(h, self.ln2, self.eps), h)

    def step(self, x: mx.array, slots: list, offsets: list, telem=None) -> mx.array:
        """One decode position for B rows (x [B, 1, D]); `slots[i]` / `offsets[i]` belong to row i."""
        h = x + self.block.step(mx.fast.rms_norm(x, self.ln1, self.eps), slots, offsets)
        return self.moe.with_residual(mx.fast.rms_norm(h, self.ln2, self.eps), h, telem, rows=True)


class Qwen3_5MoeForCausalLM:
    """The whole language model over a weights dict (names without the `language_model.` prefix)."""

    def __init__(self, cfg: Qwen3_5MoeConfig, w: dict, pool=None, fuse: frozenset = FUSIONS):
        self.cfg = cfg
        gs = cfg.q_group_size
        self.paths = KernelChoice()
        self.strategies = Strategies()
        self.pool = pool
        self.offloaded = pool is not None
        self.embed = QEmbedding(
            w["model.embed_tokens.weight"],
            w["model.embed_tokens.scales"],
            w.get("model.embed_tokens.biases"),
            gs,
        )
        self.layers = [
            Qwen3_5MoeDecoderLayer(cfg, w, i, pool=pool, fuse=fuse)
            for i in range(cfg.num_hidden_layers)
        ]
        self.norm_w = w["model.norm.weight"]
        self.eps = cfg.rms_norm_eps
        if "lm_head.weight" in w:
            self.lm_head = QLinear(
                w["lm_head.weight"], w["lm_head.scales"], w.get("lm_head.biases"), gs
            )
        else:  # tied embeddings
            self.lm_head = self.embed
        self.calibrate()

    def bind_expert_pool(self, pool) -> None:
        """Point every MoE block at `pool` (or None during a rebuild).

        The blocks hold no other reference to pool arrays, so this is what lets the old pool's
        buffers be released.
        """
        if not self.offloaded:
            raise RuntimeError("this model was built resident; it has no expert pool to rebind")
        self.pool = pool
        for layer in self.layers:
            m = layer.moe
            m.pool = pool
            m.gate_proj = pool.gate_proj if pool is not None else None
            m.up_proj = pool.up_proj if pool is not None else None
            m.down_proj = pool.down_proj if pool is not None else None
        if pool is not None:
            for lin in (pool.gate_proj, pool.up_proj, pool.down_proj):
                lin.prescaled = self.paths.dense_prescaled

    def use_kernel_choice(self, paths: KernelChoice) -> None:
        """Run the kernels `paths` selects: the engine's choice once it has checked them."""
        self.paths = paths
        for layer in self.layers:
            layer.paths = layer.block.paths = layer.moe.paths = paths
        for lin in self.qlinears():
            lin.prescaled = paths.dense_prescaled

    def use_strategies(self, strategies: Strategies) -> None:
        """Schedule expert work as `strategies` says: the engine's options, set once at load."""
        self.strategies = strategies
        for layer in self.layers:
            layer.moe.strategies = strategies

    def qlinears(self) -> list:
        """Every QLinear the model holds (the parts of fused ones and the pool's stacks included)."""
        seen, out = set(), []

        def add(v):
            if isinstance(v, FusedLinear):
                add(v.lin)
                for p in v.parts:
                    add(p)
            elif isinstance(v, QLinear) and id(v) not in seen:
                seen.add(id(v))
                out.append(v)

        add(self.lm_head)
        for layer in self.layers:
            for part in (layer, layer.block, layer.moe):
                for v in vars(part).values():
                    add(v)
        return out

    def arrays(self) -> list:
        """Every weight array the model holds (for a final evaluation after lazy construction)."""
        seen, out = set(), []

        def add(v):
            if isinstance(v, mx.array) and id(v) not in seen:
                seen.add(id(v))
                out.append(v)
            elif isinstance(v, QLinear):
                for t in (v.weight, v.scales, v.biases):
                    add(t)
            elif isinstance(v, FusedLinear):
                add(v.lin)
                for p in v.parts:
                    add(p)

        def walk(obj):
            for name in dir(obj):
                if name.startswith("__"):
                    continue
                try:
                    v = getattr(obj, name)
                except AttributeError:  # a name `dir` lists may have no value (an unset slot)
                    continue
                add(v)

        add(self.norm_w)
        add(self.embed)
        add(self.lm_head)
        for layer in self.layers:
            for part in (layer, layer.block, layer.moe):
                walk(part)
        return out

    def dense_linears(self) -> list:
        """Every QLinear a decode step drives with a stack of independent rows."""
        out = [self.lm_head]
        for layer in self.layers:
            blk = layer.block
            if isinstance(blk, Qwen3_5MoeAttention):
                out += (
                    blk.qkv_proj.parts
                    if blk.qkv_proj is not None
                    else [blk.q_proj, blk.k_proj, blk.v_proj]
                )
                out += [blk.o_proj]
            else:
                out += (
                    blk.in_proj.parts
                    if blk.in_proj is not None
                    else [blk.in_proj_qkv, blk.in_proj_z, blk.in_proj_b, blk.in_proj_a]
                )
                out += [blk.out_proj]
            m = layer.moe
            out += m.router.parts if m.router is not None else [m.gate, m.shared_gate]
            out += (
                m.shared_gu.parts
                if m.shared_gu is not None
                else [m.shared_gate_proj, m.shared_up_proj]
            )
            out += [m.shared_down_proj]
        return out

    def fused_linears(self) -> list:
        """Every FusedLinear's fused buffer, which the one-row path uses and so must calibrate."""
        out = []
        for layer in self.layers:
            blk = layer.block
            for holder in (
                getattr(blk, "qkv_proj", None),
                getattr(blk, "in_proj", None),
                layer.moe.router,
                layer.moe.shared_gu,
            ):
                if isinstance(holder, FusedLinear):
                    out.append(holder.lin)
        return out

    def calibrate(self) -> dict:
        """Measure the row-exact group size of every distinct dense weight shape (see linear.py)."""
        return calibrate_row_limits(self.dense_linears(), solo_extra=self.fused_linears())

    def forward(
        self, tokens: mx.array, state: RequestState, logits: str = "last"
    ) -> mx.array | None:
        """tokens [1, S] int32; advances `state` by S positions.

        logits: "none" (returns None), "last" ([1, V]) or "all" ([1, S, V]).
        """
        offset = state.tokens
        h = self.embed.lookup(tokens)
        for layer, slot in zip(self.layers, state.layers):
            h = layer(h, slot, offset)
        state.tokens += tokens.shape[1]
        if logits == "none":
            return None
        if logits == "last":
            h = h[:, -1, :]
        h = mx.fast.rms_norm(h, self.norm_w, self.eps)
        return self.lm_head(h)

    def forward_layer_major(
        self, head: list, bounds: list, state: RequestState, capture=()
    ) -> dict:
        """Layer-major prompt pass over consecutive chunks `bounds` starting at `state.tokens`.

        Returns {p: per-layer snapshot parts} for each chunk end p in `capture`. Bit-identical to
        `forward` chunk by chunk, since each layer state sees the same calls in the same order;
        offloaded experts are streamed once per prompt instead of once per chunk.
        """
        from jitengine_qwen3_6_35b_a3b_m2.kvcache.kv_cache import _copy_prefix
        from jitengine_qwen3_6_35b_a3b_m2.kvcache.linear_state import _materialize

        capture, parts, hs = self._embed_chunks(head, bounds, state, capture)
        self._run_layers(bounds, state, capture, parts, hs, _materialize)
        state.tokens = bounds[-1][1]
        self._capture_kv(state, capture, parts, _copy_prefix)
        return parts

    def _embed_chunks(self, head: list, bounds: list, state: RequestState, capture) -> tuple:
        """Check the bounds start at the state's position; return the capture set, empty snapshot parts and each chunk's embeddings."""
        if bounds[0][0] != state.tokens:
            raise ValueError("bounds must start at the state's position")
        capture = set(capture)
        parts = {p: [None] * len(self.layers) for p in capture}
        hs = [self.embed.lookup(mx.array(head[s:e], dtype=mx.int32)[None]) for s, e in bounds]
        return capture, parts, hs

    def _run_layers(
        self, bounds: list, state: RequestState, capture: set, parts: dict, hs: list, _materialize
    ) -> None:
        """Run every layer over all chunks, merging the routed-expert gather where each chunk allows, and record the capture points' delta states."""
        for i, (layer, slot) in enumerate(zip(self.layers, state.layers)):
            merged = self._merge_ok(bounds, layer)
            if not merged:
                self._run_chunks_solo(bounds, hs, layer, slot, capture, parts, i, _materialize)
                continue
            pre, xs, mids = [], [], []
            self._run_chunks_merged(
                bounds, hs, layer, slot, capture, parts, i, pre, xs, mids, _materialize
            )
            es = layer.moe.prefill_experts_merged(xs, [p[0] for p in pre])
            self._combine_chunks(bounds, es, pre, mids, hs, layer)
            mx.eval(*hs)

    def _merge_ok(self, bounds: list, layer) -> bool:
        """Whether the layer merges the chunks' routed-expert gather: only if each chunk alone would use stock's gather_qmm_rhs."""
        # merge only if each chunk alone would use stock's gather_qmm_rhs (same arithmetic)
        k = layer.moe.top_k
        merged = (
            self.strategies.merge_prefill_experts
            and len(bounds) > 1
            and all(layer.moe.merged_ok((e - s) * k) for s, e in bounds)
        )
        return merged

    def _run_chunks_solo(
        self, bounds: list, hs: list, layer, slot, capture: set, parts: dict, i: int, _materialize
    ) -> None:
        """Run the layer over each chunk on its own and record the capture points' delta states."""
        for c, (s, e) in enumerate(bounds):
            hs[c] = layer(hs[c], slot, s)
            arrs = [slot.keys, slot.values] if layer.is_attention else [slot.conv, slot.ssm]
            mx.eval(hs[c], *arrs)
            if e in capture and not layer.is_attention:
                parts[e][i] = ("delta", _materialize(slot.conv), _materialize(slot.ssm))
                mx.eval(parts[e][i][1], parts[e][i][2])

    def _run_chunks_merged(
        self,
        bounds: list,
        hs: list,
        layer,
        slot,
        capture: set,
        parts: dict,
        i: int,
        pre: list,
        xs: list,
        mids: list,
        _materialize,
    ) -> None:
        """Run each chunk through attention or delta and the router, leaving the routed experts for one merged gather."""
        for c, (s, e) in enumerate(bounds):
            h = hs[c] + layer.block(mx.fast.rms_norm(hs[c], layer.ln1, layer.eps), slot, s)
            xm = mx.fast.rms_norm(h, layer.ln2, layer.eps)
            pre.append(layer.moe.prefill_parts(xm))
            xs.append(xm)
            mids.append(h)
            arrs = [slot.keys, slot.values] if layer.is_attention else [slot.conv, slot.ssm]
            mx.eval(h, xm, *pre[-1], *arrs)
            if e in capture and not layer.is_attention:
                parts[e][i] = ("delta", _materialize(slot.conv), _materialize(slot.ssm))
                mx.eval(parts[e][i][1], parts[e][i][2])

    def _combine_chunks(
        self, bounds: list, es: list, pre: list, mids: list, hs: list, layer
    ) -> None:
        """Add each chunk's routed and shared expert output to its residual."""
        for c in range(len(bounds)):
            _inds, scores, sg_logit, shared = pre[c]
            hs[c] = mids[c] + layer.moe.prefill_combine(es[c], scores, sg_logit, shared)

    def _capture_kv(self, state: RequestState, capture: set, parts: dict, _copy_prefix) -> None:
        """Record each attention layer's KV prefix at every capture point."""
        for p in capture:
            for i, (layer, slot) in enumerate(zip(self.layers, state.layers)):
                if layer.is_attention:
                    parts[p][i] = ("kv", _copy_prefix(slot.keys, p), _copy_prefix(slot.values, p))

    def forward_layer_major_multi(self, jobs: list) -> list:
        """`forward_layer_major` for several prompts: jobs = [(head, bounds, state, capture)].

        Eligible chunks of every prompt share one merged expert gather per layer, with the same
        values as `forward`. Returns one {p: per-layer snapshot parts} per job.
        """
        from jitengine_qwen3_6_35b_a3b_m2.kvcache.kv_cache import _copy_prefix
        from jitengine_qwen3_6_35b_a3b_m2.kvcache.linear_state import _materialize

        caps, parts_all, hs_all = [], [], []
        self._embed_jobs(jobs, caps, parts_all, hs_all)
        self._run_layers_multi(jobs, caps, parts_all, hs_all, _materialize)
        self._capture_kv_multi(jobs, caps, parts_all, _copy_prefix)
        return parts_all

    def _embed_jobs(self, jobs: list, caps: list, parts_all: list, hs_all: list) -> None:
        """For each job, check its bounds start at the state's position and append its capture set, empty parts and chunk embeddings."""
        for head, bounds, state, capture in jobs:
            if bounds[0][0] != state.tokens:
                raise ValueError("bounds must start at the state's position")
            cap = set(capture)
            caps.append(cap)
            parts_all.append({p: [None] * len(self.layers) for p in cap})
            hs_all.append(
                [self.embed.lookup(mx.array(head[s:e], dtype=mx.int32)[None]) for s, e in bounds]
            )

    def _run_layers_multi(
        self, jobs: list, caps: list, parts_all: list, hs_all: list, _materialize
    ) -> None:
        """Run every layer over all jobs' chunks, merging the routed-expert gather across the eligible chunks."""
        for i, layer in enumerate(self.layers):
            k = layer.moe.top_k
            deferred = []
            self._run_job_chunks(jobs, caps, parts_all, hs_all, layer, i, k, deferred, _materialize)
            self._merge_deferred(layer, deferred, hs_all)

    def _run_job_chunks(
        self,
        jobs: list,
        caps: list,
        parts_all: list,
        hs_all: list,
        layer,
        i: int,
        k: int,
        deferred: list,
        _materialize,
    ) -> None:
        """Run one layer over every job's chunks; chunks that may merge their experts go to `deferred` after attention and the router."""
        for r, (_, bounds, state, _) in enumerate(jobs):
            slot = state.layers[i]
            hs = hs_all[r]
            for c, (s, e) in enumerate(bounds):
                arrs = [slot.keys, slot.values] if layer.is_attention else [slot.conv, slot.ssm]
                if self.strategies.merge_prefill_experts and layer.moe.merged_ok((e - s) * k):
                    h = hs[c] + layer.block(mx.fast.rms_norm(hs[c], layer.ln1, layer.eps), slot, s)
                    xm = mx.fast.rms_norm(h, layer.ln2, layer.eps)
                    pre = layer.moe.prefill_parts(xm)
                    deferred.append((r, c, xm, pre, h))
                    mx.eval(h, xm, *pre, *arrs)
                else:
                    hs[c] = layer(hs[c], slot, s)
                    mx.eval(hs[c], *arrs)
                if e in caps[r] and not layer.is_attention:
                    parts_all[r][e][i] = (
                        "delta",
                        _materialize(slot.conv),
                        _materialize(slot.ssm),
                    )
                    mx.eval(parts_all[r][e][i][1], parts_all[r][e][i][2])

    def _merge_deferred(self, layer, deferred: list, hs_all: list) -> None:
        """Gather the deferred chunks' routed experts once and add each output to its residual."""
        if deferred:
            es = layer.moe.prefill_experts_merged(
                [d[2] for d in deferred], [d[3][0] for d in deferred]
            )
            for (r, c, _, pre, h), e_ in zip(deferred, es):
                _inds, scores, sg_logit, shared = pre
                hs_all[r][c] = h + layer.moe.prefill_combine(e_, scores, sg_logit, shared)
            mx.eval(*[hs_all[r][c] for r, c, *_ in deferred])

    def _capture_kv_multi(self, jobs: list, caps: list, parts_all: list, _copy_prefix) -> None:
        """Advance each state past its prompt and record each attention layer's KV prefix at every capture point."""
        for (_, bounds, state, _), cap, parts in zip(jobs, caps, parts_all):
            state.tokens = bounds[-1][1]
            for p in cap:
                for i, (layer, slot) in enumerate(zip(self.layers, state.layers)):
                    if layer.is_attention:
                        parts[p][i] = (
                            "kv",
                            _copy_prefix(slot.keys, p),
                            _copy_prefix(slot.values, p),
                        )

    def decode_step(self, tokens: mx.array, states: list, telem=None) -> mx.array:
        """tokens [B, 1] int32, B >= 2 -> logits [B, V]; advances every state by one position.

        A lone row goes through `forward` instead, which keeps the solo decode path unchanged.
        """
        offsets = [s.tokens for s in states]
        h = self.embed.lookup(tokens)
        for li, layer in enumerate(self.layers):
            h = layer.step(h, [s.layers[li] for s in states], offsets, telem)
        for s in states:
            s.tokens += 1
        h = h[:, -1, :]
        h = mx.fast.rms_norm(h, self.norm_w, self.eps)
        return self.lm_head.rows(h)
