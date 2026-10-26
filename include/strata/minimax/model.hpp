// Native MiniMax-M3 geometry and tensor contract. No Qwen geometry defaults.
// GGUF names/shapes checked against llama.cpp 74b104c9, src/models/minimax-m3.cpp.
#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::minimax {
struct Config {
    int layers, width, heads, kv_heads, dim, rotary, dense_layers, dense_ff;
    int experts, active, expert_ff, shared, index_heads, index_dim, top_blocks, block, local;
    int context;
    float eps, theta, expert_scale;
    bool renormalize;

    static Config read(const GgufFile& f) {
        const auto* arch = f.get("general.architecture");
        if (!arch || arch->s != "minimax-m3") throw std::runtime_error("expected minimax-m3 GGUF");
        const auto* name = f.get("general.name");
        if (!name || name->s.find("M3.1") == std::string::npos)
            throw std::runtime_error("this experimental engine requires MiniMax M3.1 metadata; older M3 has a different attention schedule");
        auto num = [&](const char* key) {
            const auto* v = f.get(std::string("minimax-m3.") + key);
            if (!v || !v->is_num() || !std::isfinite(v->num()))
                throw std::runtime_error(std::string("missing/invalid MiniMax metadata: ") + key);
            return v->num();
        };
        auto integer = [&](const char* key, int floor = 1) {
            const double n = num(key);
            if (n < floor || n > std::numeric_limits<int>::max() || n != std::floor(n))
                throw std::runtime_error(std::string("invalid MiniMax integer: ") + key);
            return int(n);
        };
        Config c{};
        c.layers = integer("block_count"); c.width = integer("embedding_length");
        c.heads = integer("attention.head_count"); c.kv_heads = integer("attention.head_count_kv");
        c.dim = integer("attention.key_length"); c.rotary = integer("rope.dimension_count");
        c.dense_layers = integer("leading_dense_block_count", 0); c.dense_ff = integer("feed_forward_length");
        c.experts = integer("expert_count"); c.active = integer("expert_used_count");
        c.expert_ff = integer("expert_feed_forward_length"); c.shared = integer("expert_shared_count");
        c.index_heads = integer("attention.indexer.head_count"); c.index_dim = integer("attention.indexer.key_length");
        c.top_blocks = integer("attention.indexer.top_k"); c.block = integer("attention.indexer.block_size");
        c.local = integer("attention.indexer.local_blocks", 0); c.context = integer("context_length");
        c.eps = float(num("attention.layer_norm_rms_epsilon")); c.theta = float(num("rope.freq_base"));
        c.expert_scale = float(num("expert_weights_scale")); c.renormalize = num("expert_weights_norm") != 0;
        if (integer("attention.value_length") != c.dim || integer("expert_gating_func") != 2 ||
            c.heads % c.kv_heads || c.index_heads != c.kv_heads || c.rotary > c.dim ||
            c.rotary > c.index_dim || c.rotary % 2 || c.active > c.experts ||
            c.dense_layers > c.layers || c.local > c.top_blocks || c.eps <= 0 || c.theta <= 0 ||
            c.expert_scale <= 0)
            throw std::runtime_error("unsupported/inconsistent MiniMax geometry or gating");
        return c;
    }
};

struct TensorWant { std::string name; std::vector<uint64_t> shape; };
inline std::vector<TensorWant> tensor_contract(const Config& c, uint64_t vocab) {
    using V = std::vector<uint64_t>;
    const uint64_t N = c.width, D = c.dim, Q = c.heads * D, K = c.kv_heads * D;
    const uint64_t F = c.expert_ff, S = F * c.shared, I = c.index_dim;
    std::vector<TensorWant> w{{"token_embd.weight", {N, vocab}}, {"output_norm.weight", {N}},
                              {"output.weight", {N, vocab}}};
    for (int l = 0; l < c.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        auto add = [&](const char* name, V shape) { w.push_back({p + name, std::move(shape)}); };
        add("attn_norm.weight", {N}); add("attn_q.weight", {N, Q});
        add("attn_k.weight", {N, K}); add("attn_v.weight", {N, K});
        add("attn_output.weight", {Q, N}); add("attn_q_norm.weight", {D}); add("attn_k_norm.weight", {D});
        add("ffn_norm.weight", {N});
        if (l < c.dense_layers) {
            add("ffn_gate.weight", {N, uint64_t(c.dense_ff)}); add("ffn_up.weight", {N, uint64_t(c.dense_ff)});
            add("ffn_down.weight", {uint64_t(c.dense_ff), N});
        } else {
            add("ffn_gate_inp.weight", {N, uint64_t(c.experts)});
            add("exp_probs_b.bias", {uint64_t(c.experts)});
            add("ffn_gate_exps.weight", {N, F, uint64_t(c.experts)});
            add("ffn_up_exps.weight", {N, F, uint64_t(c.experts)});
            add("ffn_down_exps.weight", {F, N, uint64_t(c.experts)});
            add("ffn_gate_shexp.weight", {N, S}); add("ffn_up_shexp.weight", {N, S}); add("ffn_down_shexp.weight", {S, N});
        }
        // M3.1 has sparse attention in ALL layers. Leading dense refers only
        // to the FFN; reusing M3's first-three-attention exception is wrong.
        add("indexer.q_proj.weight", {N, I * c.index_heads}); add("indexer.k_proj.weight", {N, I});
        add("indexer.q_norm.weight", {I}); add("indexer.k_norm.weight", {I});
    }
    return w;
}

inline uint64_t validate(const GgufModel& model, const Config& c) {
    const auto* e = model.find("token_embd.weight");
    if (!e || e->shape.size() != 2 || e->shape[1] == 0) throw std::runtime_error("missing MiniMax embedding");
    for (const auto& want : tensor_contract(c, e->shape[1])) {
        size_t shard = 0;
        const auto* got = model.find(want.name, &shard);
        if (!got || got->shape != want.shape || !model.in_bounds(*got, shard))
            throw std::runtime_error("missing, malformed or truncated MiniMax tensor: " + want.name);
    }
    return e->shape[1];
}
} // namespace strata::minimax
