// Experimental native MiniMax text engine. Uses Strata's GGUF mapping and GPU
// dequantizers; no llama runtime. FP32 products are the initial correctness path.
#include "strata/minimax/model.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>

namespace M = strata::minimax;
namespace K = strata::kernels;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void cb(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("MiniMax cuBLAS error " + std::to_string(int(e)));
}

template<bool MAX> __device__ float reduce(float x) {
    __shared__ float sh[256];
    sh[threadIdx.x] = x; __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (threadIdx.x < s) sh[threadIdx.x] = MAX ? fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]) :
                                                   sh[threadIdx.x] + sh[threadIdx.x + s];
        __syncthreads();
    }
    return sh[0];
}
__global__ void convert(const uint8_t* src, float* out, size_t n, int ty) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ty == 0) out[i] = reinterpret_cast<const float*>(src)[i];
    else if (ty == 1) out[i] = __half2float(reinterpret_cast<const half*>(src)[i]);
    else out[i] = __uint_as_float(uint32_t(reinterpret_cast<const uint16_t*>(src)[i]) << 16);
}
__global__ void norm(const float* in, const float* weight, float* out, int dim, float eps) {
    in += size_t(blockIdx.x) * dim; out += size_t(blockIdx.x) * dim;
    float sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sum += in[i] * in[i];
    const float inv = rsqrtf(reduce<false>(sum) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) out[i] = in[i] * inv * weight[i];
}
__global__ void rope(float* x, int dim, int rotary, int pos, float theta, int heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * rotary / 2) return;
    const int h = i / (rotary / 2), d = i % (rotary / 2);
    const float a = pos * powf(theta, -2.0f * d / rotary), s = sinf(a), c = cosf(a);
    const int j = h * dim + d;
    const float u = x[j], v = x[j + rotary / 2];
    x[j] = u * c - v * s; x[j + rotary / 2] = u * s + v * c;
}
__global__ void swiglu(const float* gate, const float* up, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = fminf(gate[i], 7.0f), u = fmaxf(-7.0f, fminf(up[i], 7.0f));
        out[i] = g / (1.0f + expf(-1.702f * g)) * (u + 1.0f);
    }
}
__global__ void index_scores(const float* q, const float* keys, float* scores, int dim, int pos, int block, int nb) {
    const int b = blockIdx.x, h = blockIdx.y;
    float best = -INFINITY;
    for (int t = b * block + threadIdx.x; t < (b + 1) * block && t <= pos; t += blockDim.x) {
        float dot = 0;
        for (int d = 0; d < dim; ++d) dot += q[h * dim + d] * keys[size_t(t) * dim + d];
        best = fmaxf(best, dot);
    }
    best = reduce<true>(best);
    if (threadIdx.x == 0) scores[h * nb + b] = best;
}
__global__ void select_blocks(const float* scores, uint8_t* mask, int nb, int top, int local) {
    const int h = blockIdx.x;
    if (threadIdx.x) return;
    for (int b = 0; b < nb; ++b) mask[h * nb + b] = 0;
    for (int pick = 0; pick < min(nb, top); ++pick) {
        int best = -1; float score = -INFINITY;
        for (int b = 0; b < nb; ++b) {
            const float value = scores[h * nb + b] + (b >= nb - local ? 1e30f : 0.0f);
            if (!mask[h * nb + b] && (best < 0 || value > score)) { best = b; score = value; }
        }
        mask[h * nb + best] = 1;
    }
}
__global__ void att_scores(const float* q, const float* keys, const uint8_t* mask, float* scores,
                           int pos, int dim, int heads, int kv_heads, int block, int nb) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int kh = h / (heads / kv_heads);
    if (mask && !mask[kh * nb + t / block]) { if (!d) scores[h * (pos + 1) + t] = -INFINITY; return; }
    float v = 0;
    for (int j = d; j < dim; j += 32) v += q[h * dim + j] * keys[(size_t(t) * kv_heads + kh) * dim + j];
    for (int s = 16; s; s /= 2) v += __shfl_down_sync(0xffffffffu, v, s);
    if (!d) scores[h * (pos + 1) + t] = v / sqrtf(float(dim));
}
__global__ void softmax(float* scores, int n) {
    scores += size_t(blockIdx.x) * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, scores[i]);
    m = reduce<true>(m);
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { scores[i] = expf(scores[i] - m); sum += scores[i]; }
    sum = reduce<false>(sum);
    for (int i = threadIdx.x; i < n; i += blockDim.x) scores[i] /= sum;
}
__global__ void att_values(const float* p, const float* values, float* out, int pos, int dim, int heads, int kv_heads) {
    const int h = blockIdx.x, kh = h / (heads / kv_heads);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0;
        for (int t = 0; t <= pos; ++t) sum += p[h * (pos + 1) + t] * values[(size_t(t) * kv_heads + kh) * dim + d];
        out[h * dim + d] = sum;
    }
}

struct Tensor { const uint8_t* data; int type, in, out; size_t row; float scale; };
struct Entry { uint8_t* ptr; size_t bytes; uint64_t used; };
struct Device {
    int id;
    cudaStream_t stream{};
    cublasHandle_t blas{};
    size_t budget, cached = 0;
    uint64_t clock = 0;
    std::map<std::pair<const uint8_t*, size_t>, Entry> cache;
    std::vector<float*> allocations;
    float *x, *h, *q, *k, *v, *iq, *ik, *att, *gate, *up, *act, *y, *mix, *router, *scores, *bs, *tile, *logits;
    uint8_t* mask{};
    Device(int gpu, const M::Config& c, int ctx, int vocab, size_t cache_bytes): id(gpu), budget(cache_bytes) {
        ck(cudaSetDevice(id)); ck(cudaStreamCreate(&stream)); cb(cublasCreate(&blas)); cb(cublasSetStream(blas, stream));
        cb(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
        const size_t ff = std::max(c.dense_ff, c.expert_ff * c.shared);
        x = alloc(c.width); h = alloc(c.width); q = alloc(c.heads * c.dim); k = alloc(c.kv_heads * c.dim);
        v = alloc(c.kv_heads * c.dim); iq = alloc(c.index_heads * c.index_dim); ik = alloc(c.index_dim);
        att = alloc(c.heads * c.dim); gate = alloc(ff); up = alloc(ff); act = alloc(ff);
        y = alloc(c.width); mix = alloc(c.width); router = alloc(c.experts); logits = alloc(vocab);
        scores = alloc(size_t(c.heads) * ctx); bs = alloc(size_t(c.index_heads) * ((ctx + c.block - 1) / c.block));
        tile = alloc(size_t(1024) * std::max({c.width, c.heads * c.dim, int(ff)}));
        ck(cudaMalloc(&mask, size_t(c.index_heads) * ((ctx + c.block - 1) / c.block)));
    }
    ~Device() {
        cudaSetDevice(id); cudaStreamSynchronize(stream);
        for (auto& e : cache) cudaFree(e.second.ptr);
        for (auto p : allocations) cudaFree(p);
        cudaFree(mask); cublasDestroy(blas); cudaStreamDestroy(stream);
    }
    float* alloc(size_t n) { float* p{}; ck(cudaMalloc(&p, n * sizeof(float))); allocations.push_back(p); return p; }
    void sync() { ck(cudaStreamSynchronize(stream)); ck(cudaGetLastError()); }
    const uint8_t* raw(const uint8_t* host, size_t bytes) {
        if (bytes > budget) throw std::runtime_error("--cache-mib is smaller than a matrix tile");
        const auto key = std::make_pair(host, bytes);
        auto found = cache.find(key);
        if (found != cache.end()) { found->second.used = ++clock; return found->second.ptr; }
        while (cached + bytes > budget) {
            auto old = std::min_element(cache.begin(), cache.end(), [](const auto& a, const auto& b) { return a.second.used < b.second.used; });
            sync(); ck(cudaFree(old->second.ptr)); cached -= old->second.bytes; cache.erase(old);
        }
        uint8_t* p{}; ck(cudaMalloc(&p, bytes)); ck(cudaMemcpyAsync(p, host, bytes, cudaMemcpyHostToDevice, stream));
        cache.emplace(key, Entry{p, bytes, ++clock}); cached += bytes;
        return p;
    }
    void decode(int type, const uint8_t* src, size_t n, float* out) {
        if (type == 0 || type == 1 || type == 30)
            convert<<<unsigned((n + 255) / 256), 256, 0, stream>>>(src, out, n, type);
        else if (K::embed_type_supported(type) && n % 256 == 0)
            K::iq_dequant_f32(type, src, n, out, stream);
        else throw std::runtime_error("MiniMax tensor type/alignment is not supported: " + std::to_string(type));
    }
    void linear(const Tensor& w, const float* in, float* out) {
        const float one = 1, zero = 0;
        for (int row = 0; row < w.out; row += 1024) {
            const int rows = std::min(1024, w.out - row);
            decode(w.type, raw(w.data + size_t(row) * w.row, size_t(rows) * w.row), size_t(rows) * w.in, tile);
            cb(cublasSgemv(blas, CUBLAS_OP_T, w.in, rows, &one, tile, w.in, in, 1, &zero, out + row, 1));
        }
        if (w.scale != 1.0f) cb(cublasSscal(blas, w.out, &w.scale, out, 1));
    }
    void add(const float* in, float* out, int n, float weight = 1) { cb(cublasSaxpy(blas, n, &weight, in, 1, out, 1)); }
};

struct Layer {
    int device;
    float *an, *qn, *kn, *fn, *iqn = nullptr, *ikn = nullptr, *kc, *vc, *ic = nullptr;
    std::vector<float> bias;
};
struct Engine {
    strata::GgufModel model;
    M::Config c;
    int vocab, ctx, position = 0;
    std::vector<std::unique_ptr<Device>> devices;
    std::vector<Layer> layers;
    float* output_norm;
    Engine(const std::string& path, const std::vector<int>& gpus, int context, size_t budget):
        model(strata::GgufModel::open(path)), c(M::Config::read(model.meta())), vocab(int(M::validate(model, c))), ctx(context) {
        if (ctx < 1 || ctx > c.context) throw std::runtime_error("context outside model limit");
        for (int gpu : gpus) devices.push_back(std::make_unique<Device>(gpu, c, ctx, vocab, budget));
        for (int l = 0; l < c.layers; ++l) {
            Layer y{}; y.device = std::min(int(devices.size()) - 1, int(int64_t(l) * devices.size() / c.layers));
            auto& d = *devices[y.device]; ck(cudaSetDevice(d.id));
            const std::string p = "blk." + std::to_string(l) + ".";
            y.an = vector(d, p + "attn_norm.weight"); y.qn = vector(d, p + "attn_q_norm.weight");
            y.kn = vector(d, p + "attn_k_norm.weight"); y.fn = vector(d, p + "ffn_norm.weight");
            y.kc = d.alloc(size_t(ctx) * c.kv_heads * c.dim); y.vc = d.alloc(size_t(ctx) * c.kv_heads * c.dim);
            y.iqn = vector(d, p + "indexer.q_norm.weight"); y.ikn = vector(d, p + "indexer.k_norm.weight");
            y.ic = d.alloc(size_t(ctx) * c.index_dim);
            if (l >= c.dense_layers) {
                float* b = vector(d, p + "exp_probs_b.bias"); y.bias.resize(c.experts);
                ck(cudaMemcpyAsync(y.bias.data(), b, c.experts * sizeof(float), cudaMemcpyDeviceToHost, d.stream));
            }
            d.sync(); layers.push_back(std::move(y));
        }
        ck(cudaSetDevice(devices[0]->id)); output_norm = vector(*devices[0], "output_norm.weight");
    }
    Tensor tensor(const std::string& name, int expert = -1) {
        size_t shard = 0; const auto* t = model.find(name, &shard);
        if (!t) throw std::runtime_error("missing tensor " + name);
        int be = 0, bb = 0;
        if (!strata::block_geometry(t->type, be, bb) || t->shape[0] % be)
            throw std::runtime_error("bad tensor block geometry " + name);
        const int in = int(t->shape[0]), out = t->shape.size() > 1 ? int(t->shape[1]) : 1;
        const size_t row = size_t(in / be) * bb;
        const uint8_t* data = model.shard(shard).tensor_data(*t);
        if (expert >= 0) {
            if (t->shape.size() != 3 || uint64_t(expert) >= t->shape[2]) throw std::runtime_error("bad expert slice");
            data += size_t(expert) * out * row;
        }
        float scale = 1.0f;
        if (t->type == 40) {
            size_t ss = 0;
            const std::string scale_name = name.substr(0, name.size() - 7) + ".scale";
            const auto* s = model.find(scale_name, &ss);
            const uint64_t count = t->shape.size() == 3 ? t->shape[2] : 1;
            if (!s || s->type != 0 || s->shape != std::vector<uint64_t>{count} || !model.in_bounds(*s, ss) ||
                (count > 1 && expert < 0)) throw std::runtime_error("missing/invalid NVFP4 scale: " + scale_name);
            std::memcpy(&scale, model.shard(ss).tensor_data(*s) + size_t(std::max(0, expert)) * sizeof(float), sizeof(float));
            if (!std::isfinite(scale) || scale <= 0) throw std::runtime_error("invalid NVFP4 scale value");
        }
        return {data, int(t->type), in, out, row, scale};
    }
    float* vector(Device& d, const std::string& name) {
        const Tensor t = tensor(name); float* p = d.alloc(t.in);
        d.decode(t.type, d.raw(t.data, t.row), t.in, p); return p;
    }
    void projection(Device& d, const std::string& name, const float* x, float* y, int expert = -1) {
        d.linear(tensor(name, expert), x, y);
    }
    void ffn(Device& d, const std::string& p, const std::string& kind, int width, int expert = -1) {
        projection(d, p + "ffn_gate" + kind + ".weight", d.h, d.gate, expert);
        projection(d, p + "ffn_up" + kind + ".weight", d.h, d.up, expert);
        swiglu<<<(width + 255) / 256, 256, 0, d.stream>>>(d.gate, d.up, d.act, width);
        projection(d, p + "ffn_down" + kind + ".weight", d.act, d.y, expert);
    }
    void transfer(Device& from, Device& to) {
        if (&from == &to) return;
        std::vector<float> tmp(c.width); from.sync();
        ck(cudaMemcpy(tmp.data(), from.x, c.width * sizeof(float), cudaMemcpyDeviceToHost));
        ck(cudaSetDevice(to.id)); ck(cudaMemcpyAsync(to.x, tmp.data(), c.width * sizeof(float), cudaMemcpyHostToDevice, to.stream)); to.sync();
    }
    std::vector<float> step(int token) {
        if (token < 0 || token >= vocab || position >= ctx) throw std::runtime_error("token or context out of range");
        auto& first = *devices[0]; ck(cudaSetDevice(first.id));
        const Tensor emb = tensor("token_embd.weight");
        first.decode(emb.type, first.raw(emb.data + size_t(token) * emb.row, emb.row), c.width, first.x);
        Device* prev = &first;
        for (int l = 0; l < c.layers; ++l) {
            Layer& y = layers[l]; auto& d = *devices[y.device]; transfer(*prev, d); ck(cudaSetDevice(d.id)); prev = &d;
            const std::string p = "blk." + std::to_string(l) + ".";
            norm<<<1, 256, 0, d.stream>>>(d.x, y.an, d.h, c.width, c.eps);
            projection(d, p + "attn_q.weight", d.h, d.q); projection(d, p + "attn_k.weight", d.h, d.k);
            projection(d, p + "attn_v.weight", d.h, d.v);
            norm<<<c.heads, 256, 0, d.stream>>>(d.q, y.qn, d.q, c.dim, c.eps);
            norm<<<c.kv_heads, 256, 0, d.stream>>>(d.k, y.kn, d.k, c.dim, c.eps);
            rope<<<(c.heads * c.rotary / 2 + 255) / 256, 256, 0, d.stream>>>(d.q, c.dim, c.rotary, position, c.theta, c.heads);
            rope<<<(c.kv_heads * c.rotary / 2 + 255) / 256, 256, 0, d.stream>>>(d.k, c.dim, c.rotary, position, c.theta, c.kv_heads);
            const size_t kv = size_t(c.kv_heads) * c.dim;
            ck(cudaMemcpyAsync(y.kc + size_t(position) * kv, d.k, kv * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
            ck(cudaMemcpyAsync(y.vc + size_t(position) * kv, d.v, kv * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
            const int nb = position / c.block + 1;
            if (y.ic) {
                projection(d, p + "indexer.q_proj.weight", d.h, d.iq); projection(d, p + "indexer.k_proj.weight", d.h, d.ik);
                norm<<<c.index_heads, 256, 0, d.stream>>>(d.iq, y.iqn, d.iq, c.index_dim, c.eps);
                norm<<<1, 256, 0, d.stream>>>(d.ik, y.ikn, d.ik, c.index_dim, c.eps);
                rope<<<(c.index_heads * c.rotary / 2 + 255) / 256, 256, 0, d.stream>>>(d.iq, c.index_dim, c.rotary, position, c.theta, c.index_heads);
                rope<<<(c.rotary / 2 + 255) / 256, 256, 0, d.stream>>>(d.ik, c.index_dim, c.rotary, position, c.theta, 1);
                ck(cudaMemcpyAsync(y.ic + size_t(position) * c.index_dim, d.ik, c.index_dim * sizeof(float), cudaMemcpyDeviceToDevice, d.stream));
                index_scores<<<dim3(nb, c.index_heads), 256, 0, d.stream>>>(d.iq, y.ic, d.bs, c.index_dim, position, c.block, nb);
                select_blocks<<<c.index_heads, 1, 0, d.stream>>>(d.bs, d.mask, nb, c.top_blocks, c.local);
            }
            att_scores<<<dim3(c.heads, position + 1), 32, 0, d.stream>>>(d.q, y.kc, y.ic ? d.mask : nullptr, d.scores,
                position, c.dim, c.heads, c.kv_heads, c.block, nb);
            softmax<<<c.heads, 256, 0, d.stream>>>(d.scores, position + 1);
            att_values<<<c.heads, 256, 0, d.stream>>>(d.scores, y.vc, d.att, position, c.dim, c.heads, c.kv_heads);
            projection(d, p + "attn_output.weight", d.att, d.y); d.add(d.y, d.x, c.width);
            norm<<<1, 256, 0, d.stream>>>(d.x, y.fn, d.h, c.width, c.eps);
            if (l < c.dense_layers) { ffn(d, p, "", c.dense_ff); d.add(d.y, d.x, c.width); }
            else {
                projection(d, p + "ffn_gate_inp.weight", d.h, d.router);
                std::vector<float> logits(c.experts), weights(c.experts);
                ck(cudaMemcpyAsync(logits.data(), d.router, c.experts * sizeof(float), cudaMemcpyDeviceToHost, d.stream)); d.sync();
                std::vector<int> ids(c.experts); std::iota(ids.begin(), ids.end(), 0);
                for (int i = 0; i < c.experts; ++i) weights[i] = 1.0f / (1.0f + std::exp(-logits[i]));
                std::partial_sort(ids.begin(), ids.begin() + c.active, ids.end(), [&](int a, int b) {
                    const float x = weights[a] + y.bias[a], z = weights[b] + y.bias[b]; return x == z ? a < b : x > z;
                });
                float total = 0; for (int i = 0; i < c.active; ++i) total += weights[ids[i]];
                if (c.renormalize && !(total > 0)) throw std::runtime_error("non-finite/zero MiniMax routing mass");
                ck(cudaMemsetAsync(d.mix, 0, c.width * sizeof(float), d.stream));
                for (int i = 0; i < c.active; ++i) {
                    const int e = ids[i]; ffn(d, p, "_exps", c.expert_ff, e);
                    d.add(d.y, d.mix, c.width, c.expert_scale * weights[e] / (c.renormalize ? total : 1.0f));
                }
                ffn(d, p, "_shexp", c.expert_ff * c.shared); d.add(d.y, d.mix, c.width); d.add(d.mix, d.x, c.width);
            }
        }
        transfer(*prev, first); ck(cudaSetDevice(first.id));
        norm<<<1, 256, 0, first.stream>>>(first.x, output_norm, first.h, c.width, c.eps);
        projection(first, "output.weight", first.h, first.logits);
        std::vector<float> out(vocab);
        ck(cudaMemcpyAsync(out.data(), first.logits, vocab * sizeof(float), cudaMemcpyDeviceToHost, first.stream)); first.sync();
        ++position;
        for (float v : out) if (!std::isfinite(v)) throw std::runtime_error("non-finite MiniMax logits");
        return out;
    }
};

static std::vector<int> ints(const std::string& text) {
    std::stringstream in(text); std::string part; std::vector<int> out;
    while (std::getline(in, part, ',')) { size_t used = 0; int x = std::stoi(part, &used); if (used != part.size()) throw std::runtime_error("invalid integer list"); out.push_back(x); }
    if (out.empty()) throw std::runtime_error("empty integer list"); return out;
}
int main(int argc, char** argv) try {
    std::string path, tokens, dump; std::vector<int> gpu{0}; int context = 4096, n = 0; size_t budget = size_t(512) << 20;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() { if (++i == argc) throw std::runtime_error("missing value for " + arg); return std::string(argv[i]); };
        if (arg == "--model") path = next(); else if (arg == "--tokens") tokens = next();
        else if (arg == "--gpu") gpu = ints(next()); else if (arg == "--context") context = std::stoi(next());
        else if (arg == "--n") n = std::stoi(next()); else if (arg == "--dump-all-logits") dump = next();
        else if (arg == "--cache-mib") { const int mib = std::stoi(next()); if (mib < 64) throw std::runtime_error("cache must be >=64 MiB"); budget = size_t(mib) << 20; }
        else if (arg == "--help") {
            std::cout << "Experimental native MiniMax text engine.\n--model SHARD --tokens ID,ID [--n N] [--gpu 0,1] [--context N] [--cache-mib N] [--dump-all-logits FILE]\n"; return 0;
        } else throw std::runtime_error("unknown option " + arg);
    }
    if (path.empty() || tokens.empty() || n < 0) throw std::runtime_error("--model and --tokens are required; --n must be nonnegative");
    auto sorted = gpu; std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) throw std::runtime_error("invalid GPU list");
    const auto ids = ints(tokens); if (size_t(context) < ids.size() + size_t(n)) throw std::runtime_error("context too small for prompt plus output");
    Engine e(path, gpu, context, budget); std::ofstream file;
    if (!dump.empty()) { file.open(dump, std::ios::binary); if (!file) throw std::runtime_error("cannot open logits dump"); }
    std::vector<float> logits;
    for (int id : ids) { logits = e.step(id); if (file.is_open()) { file.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float)); if (!file) throw std::runtime_error("logits dump write failed"); } }
    const auto* eos = e.model.meta().get("tokenizer.ggml.eos_token_id");
    for (int i = 0; i < n; ++i) {
        const int token = int(std::max_element(logits.begin(), logits.end()) - logits.begin()); std::cout << "T " << token << std::endl;
        if (eos && token == int(eos->u)) break;
        if (i + 1 < n) logits = e.step(token);
    }
    return 0;
} catch (const std::exception& e) { std::cerr << "strata-minimax: " << e.what() << '\n'; return 1; }
