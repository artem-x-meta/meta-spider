// M2b: ggml forward of the Doubter SelectiveEncoder (standalone, no llama).
// Loads doubter_sidecar.gguf (encoder_type=selective), reads acts.bin [NL×HID], builds a graph
// per the numpy spec (meta_deploy/spec.py::selective_forward), writes cog [NL×HID] → out.bin.
//
// Sizes — GENERIC (phase 3, not hardcoded): NL from meta_spider.num_cognitive_tokens,
// HID/BN from the shape of enc.layer_projectors.0.1.weight (ne=[HID, BN]).
//
// Spec (= spec.selective_forward, pinned by a test vs PyTorch, diff<1e-4):
//   projector_i: LN(eps1e-5) + Linear(HID→BN) + GELU(erf);  * tanh(layer_gates.i) [scalar]
//   stack NL → [BN, NL]
//   output_proj: LN + Linear(BN→HID) + GELU(erf) + Linear(HID→HID);  output_norm(LN)

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const float EPS = 1e-5f;
static struct ggml_context * WCTX = nullptr;   // weights (from gguf, with data)

static ggml_tensor * W(const char * name) {
    ggml_tensor * t = ggml_get_tensor(WCTX, name);
    if (!t) { fprintf(stderr, "MISSING tensor: %s\n", name); exit(1); }
    return t;
}
static ggml_tensor * Wf(const char * fmt, int i) {
    char buf[256]; snprintf(buf, sizeof(buf), fmt, i); return W(buf);
}
// LayerNorm (normalization over ne0) with affine parameters
static ggml_tensor * ln(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(c, x, EPS);
    x = ggml_mul(c, x, w);
    x = ggml_add(c, x, b);
    return x;
}
// Linear: y = x @ Wᵀ + b   (W in ggml ne=[in, out])
static ggml_tensor * lin(ggml_context * c, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    x = ggml_mul_mat(c, w, x);
    if (b) x = ggml_add(c, x, b);
    return x;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s sidecar.gguf acts.bin out_cpp.bin\n", argv[0]);
        return 1;
    }
    const char * sidecar = argv[1];
    const char * acts_path = argv[2];
    const char * out_path = argv[3];

    // --- load weights from gguf (with data) ---
    struct gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &WCTX };
    struct gguf_context * gguf = gguf_init_from_file(sidecar, gp);
    if (!gguf) { fprintf(stderr, "failed to open %s\n", sidecar); return 1; }

    // --- sizes: NL from metadata, HID/BN from the projector shape (generic) ---
    int64_t k_nl = gguf_find_key(gguf, "meta_spider.num_cognitive_tokens");
    if (k_nl < 0) { fprintf(stderr, "no meta_spider.num_cognitive_tokens\n"); return 1; }
    const int NL = (int) gguf_get_val_u32(gguf, k_nl);
    ggml_tensor * p0w = W("enc.layer_projectors.0.1.weight");   // ne=[HID, BN]
    const int HID = (int) p0w->ne[0];
    const int BN  = (int) p0w->ne[1];
    fprintf(stderr, "selective: NL=%d HID=%d BN=%d  tensors=%lld\n", NL, HID, BN,
            (long long) gguf_get_n_tensors(gguf));

    // --- input: activations [NL, HID] (layer by layer) ---
    std::vector<float> acts((size_t) NL * HID);
    FILE * fa = fopen(acts_path, "rb");
    if (!fa || fread(acts.data(), sizeof(float), acts.size(), fa) != acts.size()) {
        fprintf(stderr, "failed to read %s\n", acts_path); return 1;
    }
    fclose(fa);

    // --- compute context (no_alloc, the graph is allocated by gallocr) ---
    size_t mem = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    struct ggml_init_params cp = { mem, nullptr, /*no_alloc*/ true };
    struct ggml_context * c = ggml_init(cp);

    ggml_tensor * in_acts = ggml_new_tensor_2d(c, GGML_TYPE_F32, HID, NL);  // ne0=HID, ne1=NL
    ggml_set_name(in_acts, "in_acts");
    ggml_set_input(in_acts);

    // --- per-layer projectors + per-layer scalar gate → stacked[BN, NL] ---
    ggml_tensor * stacked = nullptr;
    for (int i = 0; i < NL; ++i) {
        ggml_tensor * a = ggml_view_2d(c, in_acts, HID, 1, in_acts->nb[1], (size_t) i * in_acts->nb[1]);
        a = ggml_cont(c, a);
        ggml_tensor * p = ln(c, a, Wf("enc.layer_projectors.%d.0.weight", i),
                                   Wf("enc.layer_projectors.%d.0.bias", i));
        p = lin(c, Wf("enc.layer_projectors.%d.1.weight", i),
                   Wf("enc.layer_projectors.%d.1.bias", i), p);   // [BN, 1]
        p = ggml_gelu_erf(c, p);
        // scalar gate tanh(layer_gates.i): WCTX holds the data → read from host
        const float graw = ((const float *) Wf("enc.layer_gates.%d", i)->data)[0];
        p = ggml_scale(c, p, tanhf(graw));
        stacked = stacked ? ggml_concat(c, stacked, p, 1) : p;    // → [BN, NL]
    }

    // --- output_proj (per-token) + output_norm ---
    ggml_tensor * x = ln(c, stacked, W("enc.output_proj.0.weight"), W("enc.output_proj.0.bias"));
    x = lin(c, W("enc.output_proj.1.weight"), W("enc.output_proj.1.bias"), x);  // [HID, NL]
    x = ggml_gelu_erf(c, x);
    x = lin(c, W("enc.output_proj.3.weight"), W("enc.output_proj.3.bias"), x);  // [HID, NL]
    x = ln(c, x, W("enc.output_norm.weight"), W("enc.output_norm.bias"));
    ggml_set_name(x, "cog");
    ggml_set_output(x);

    // --- build graph + allocate + compute ---
    struct ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, x);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "graph alloc fail\n"); return 1; }

    ggml_backend_tensor_set(in_acts, acts.data(), 0, acts.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute fail\n"); return 1;
    }

    // --- output cog [HID, NL] → file (order: token0[HID], token1[HID], ...) ---
    std::vector<float> cog((size_t) NL * HID);
    ggml_backend_tensor_get(x, cog.data(), 0, cog.size() * sizeof(float));
    FILE * fo = fopen(out_path, "wb");
    fwrite(cog.data(), sizeof(float), cog.size(), fo);
    fclose(fo);
    fprintf(stderr, "→ %s  (cog %d×%d)\n", out_path, NL, HID);

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(c);
    gguf_free(gguf);
    ggml_free(WCTX);
    return 0;
}
