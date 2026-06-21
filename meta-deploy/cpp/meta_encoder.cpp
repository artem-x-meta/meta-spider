// M2b: ggml forward of the Doubter multi_token encoder (standalone, no llama).
// Loads doubter_sidecar.gguf, reads enc_testcase_acts.bin [num_layers×hidden],
// builds a graph per the numpy spec (validate_encoder_numpy.py), writes cog → enc_testcase_cpp.bin.
// Comparison with the PyTorch reference — compare_cog.py.
//
// Spec (pinned in M2a, diff 1e-6):
//   projector_i: LN(eps1e-5) + Linear(2304→256) + GELU(erf)
//   stack 5 → kv[256,5];  queries[256,8]
//   MHA(8 heads×32, scale 1/√32): Q from queries, K/V from kv, in_proj split [Wq;Wk;Wv], out_proj
//   output_proj: LN + Linear(256→2304) + GELU(erf) + Linear(2304→2304);  output_norm

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// --- sidecar sizes (Gemma-2B v9, multi_token) ---
static const int HID   = 2304;
static const int BN    = 256;   // bottleneck
static const int NCOG  = 8;     // cognitive tokens
static const int NHEAD = 8;
static const int HD    = BN / NHEAD;  // 32
static const int NL    = 5;     // target layers
static const float EPS = 1e-5f;

static struct ggml_context * WCTX = nullptr;  // weights (from gguf, with data)

static ggml_tensor * W(const char * name) {
    ggml_tensor * t = ggml_get_tensor(WCTX, name);
    if (!t) { fprintf(stderr, "MISSING tensor: %s\n", name); exit(1); }
    return t;
}
static ggml_tensor * Wf(struct ggml_context * c, const char * fmt, int i) {
    char buf[256]; snprintf(buf, sizeof(buf), fmt, i); return W(buf);
}

// LayerNorm with affine parameters (normalization over ne0)
static ggml_tensor * ln(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(c, x, EPS);
    x = ggml_mul(c, x, w);
    x = ggml_add(c, x, b);
    return x;
}
// Linear: y = x @ Wᵀ + b   (W in ggml ne=[in,out])
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
    fprintf(stderr, "tensors loaded: %lld\n", (long long) gguf_get_n_tensors(gguf));

    // --- read input activations [NL, HID] ---
    std::vector<float> acts(NL * HID);
    FILE * fa = fopen(acts_path, "rb");
    if (!fa || fread(acts.data(), sizeof(float), acts.size(), fa) != acts.size()) {
        fprintf(stderr, "failed to read %s\n", acts_path); return 1;
    }
    fclose(fa);

    // --- compute context (no_alloc, the graph is allocated by gallocr) ---
    size_t mem = ggml_tensor_overhead() * 2048 + ggml_graph_overhead();
    struct ggml_init_params cp = { mem, nullptr, /*no_alloc*/ true };
    struct ggml_context * c = ggml_init(cp);

    // input: activations as [HID, NL] (ne0=HID, ne1=NL)
    ggml_tensor * in_acts = ggml_new_tensor_2d(c, GGML_TYPE_F32, HID, NL);
    ggml_set_name(in_acts, "in_acts");
    ggml_set_input(in_acts);

    // --- per-layer projectors → kv[BN, NL] ---
    ggml_tensor * kv = nullptr;
    for (int i = 0; i < NL; ++i) {
        // column i: act_i [HID]
        ggml_tensor * a = ggml_view_2d(c, in_acts, HID, 1, in_acts->nb[1], (size_t) i * in_acts->nb[1]);
        a = ggml_cont(c, a);
        ggml_tensor * p = ln(c, a, Wf(c, "enc.layer_projectors.%d.0.weight", i),
                                   Wf(c, "enc.layer_projectors.%d.0.bias", i));
        p = lin(c, Wf(c, "enc.layer_projectors.%d.1.weight", i),
                   Wf(c, "enc.layer_projectors.%d.1.bias", i), p);   // [BN,1]
        p = ggml_gelu_erf(c, p);
        kv = kv ? ggml_concat(c, kv, p, 1) : p;   // → [BN, NL]
    }

    // --- cross-attention: queries × kv ---
    ggml_tensor * q_in = W("enc.queries");                 // [BN, NCOG]
    ggml_tensor * in_w = W("enc.cross_attn.in_proj_weight"); // ne=[BN, 3*BN]
    ggml_tensor * in_b = W("enc.cross_attn.in_proj_bias");   // [3*BN]
    ggml_tensor * Wq = ggml_view_2d(c, in_w, BN, BN, in_w->nb[1], (size_t) 0 * BN * in_w->nb[1]);
    ggml_tensor * Wk = ggml_view_2d(c, in_w, BN, BN, in_w->nb[1], (size_t) 1 * BN * in_w->nb[1]);
    ggml_tensor * Wv = ggml_view_2d(c, in_w, BN, BN, in_w->nb[1], (size_t) 2 * BN * in_w->nb[1]);
    ggml_tensor * bq = ggml_view_1d(c, in_b, BN, (size_t) 0 * BN * in_b->nb[0]);
    ggml_tensor * bk = ggml_view_1d(c, in_b, BN, (size_t) 1 * BN * in_b->nb[0]);
    ggml_tensor * bv = ggml_view_1d(c, in_b, BN, (size_t) 2 * BN * in_b->nb[0]);

    ggml_tensor * Q = ggml_add(c, ggml_mul_mat(c, ggml_cont(c, Wq), q_in), bq); // [BN, NCOG]
    ggml_tensor * K = ggml_add(c, ggml_mul_mat(c, ggml_cont(c, Wk), kv),   bk); // [BN, NL]
    ggml_tensor * V = ggml_add(c, ggml_mul_mat(c, ggml_cont(c, Wv), kv),   bv); // [BN, NL]

    // heads: [HD, NHEAD, n] → permute [HD, n, NHEAD]
    Q = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, Q, HD, NHEAD, NCOG), 0, 2, 1, 3)); // [HD,NCOG,NHEAD]
    K = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, K, HD, NHEAD, NL),   0, 2, 1, 3)); // [HD,NL,NHEAD]
    ggml_tensor * kq = ggml_mul_mat(c, K, Q);                       // [NL, NCOG, NHEAD]
    kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / sqrtf((float) HD), 0.0f);
    // V → [NL, HD, NHEAD]
    ggml_tensor * Vp = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, V, HD, NHEAD, NL), 1, 2, 0, 3));
    ggml_tensor * kqv = ggml_mul_mat(c, Vp, kq);                    // [HD, NCOG, NHEAD]
    kqv = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3));          // [HD, NHEAD, NCOG]
    ggml_tensor * att = ggml_reshape_2d(c, kqv, BN, NCOG);          // [BN, NCOG]
    att = lin(c, W("enc.cross_attn.out_proj.weight"), W("enc.cross_attn.out_proj.bias"), att);

    // --- output_proj + output_norm ---
    ggml_tensor * x = ln(c, att, W("enc.output_proj.0.weight"), W("enc.output_proj.0.bias"));
    x = lin(c, W("enc.output_proj.1.weight"), W("enc.output_proj.1.bias"), x);  // [HID, NCOG]
    x = ggml_gelu_erf(c, x);
    x = lin(c, W("enc.output_proj.3.weight"), W("enc.output_proj.3.bias"), x);  // [HID, NCOG]
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

    // --- output cog [HID, NCOG] → file (order: token0[HID], token1[HID], ...) ---
    std::vector<float> cog(NCOG * HID);
    ggml_backend_tensor_get(x, cog.data(), 0, cog.size() * sizeof(float));
    FILE * fo = fopen(out_path, "wb");
    fwrite(cog.data(), sizeof(float), cog.size(), fo);
    fclose(fo);
    fprintf(stderr, "→ %s  (cog %d×%d)\n", out_path, NCOG, HID);

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(c);
    gguf_free(gguf);
    ggml_free(WCTX);
    return 0;
}
