// GoalAnchor TRANSFORMER-encoder forward in ggml (standalone, no llama).
// Mirrors meta_core.encoders.transformer.TransformerEncoder bit-for-bit (validate 1e-5).
//
// Loads anchor_sidecar.gguf (enc.* tensors + shape metadata), reads anchor_acts.bin [NL×HID],
// builds the encoder graph, writes cog → out.bin. Compare with validate_anchor_encoder.py's ref.
//
// Graph (per-layer proj → +pos → input_norm → NB×block(self-attn+FFN) → output_proj → output_norm):
//   projector_i: Linear(HID→ED, no bias)         [enc.layer_projectors.i.weight]
//   + layer_pos_embed[ED,NL]                      [enc.layer_pos_embed]
//   input_norm: LN(ED)                            [enc.input_norm.{weight,bias}]
//   block b (pre-norm, no bias, no gates):
//     h = LN(x)[norm_attn]; Q/K/V = Linear(ED→ED); MHA(NH heads, 1/√HD); o_proj; x += att
//     h = LN(x)[norm_ffn];  ffn.0(ED→FFN·ED) GELU ffn.2(FFN·ED→ED); x += h
//   output_proj: LN(ED)[.0] + Linear(ED→HID)[.1] + GELU + Linear(HID→HID)[.3]
//   output_norm: LN(HID)                          [enc.output_norm.{weight,bias}]

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const float EPS = 1e-5f;
static struct ggml_context * WCTX = nullptr;

static ggml_tensor * W(const char * name) {
    ggml_tensor * t = ggml_get_tensor(WCTX, name);
    if (!t) { fprintf(stderr, "MISSING tensor: %s\n", name); exit(1); }
    return t;
}
static ggml_tensor * Wf(const char * fmt, int i) {
    char b[256]; snprintf(b, sizeof(b), fmt, i); return W(b);
}
static ggml_tensor * Wf2(const char * fmt, int a, const char * s) {
    char b[256]; snprintf(b, sizeof(b), fmt, a, s); return W(b);
}
static uint32_t meta_u32(gguf_context * g, const char * key, uint32_t def) {
    int64_t k = gguf_find_key(g, key);
    return k < 0 ? def : gguf_get_val_u32(g, k);
}

// LayerNorm (over ne0) with affine
static ggml_tensor * ln(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(c, x, EPS); x = ggml_mul(c, x, w); x = ggml_add(c, x, b); return x;
}
// Linear y = W @ x  (+b), W ggml ne=[in,out]
static ggml_tensor * lin(ggml_context * c, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    x = ggml_mul_mat(c, ggml_cont(c, w), x); if (b) x = ggml_add(c, x, b); return x;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s sidecar.gguf acts.bin out.bin\n", argv[0]); return 1; }
    const char * sidecar = argv[1], * acts_path = argv[2], * out_path = argv[3];

    struct gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &WCTX };
    struct gguf_context * gguf = gguf_init_from_file(sidecar, gp);
    if (!gguf) { fprintf(stderr, "failed to open %s\n", sidecar); return 1; }

    const int HID = (int) meta_u32(gguf, "meta_spider.hidden_dim", 0);
    const int NL  = (int) meta_u32(gguf, "meta_spider.num_cognitive_tokens", 0);
    const int ED  = (int) meta_u32(gguf, "meta_spider.encoder_dim", 0);
    const int NB  = (int) meta_u32(gguf, "meta_spider.encoder_num_blocks", 0);
    const int NH  = (int) meta_u32(gguf, "meta_spider.enc_num_heads", 0);
    const int FFNX = (int) meta_u32(gguf, "meta_spider.encoder_ffn_expansion", 4);
    const int USE_POS = (int) meta_u32(gguf, "meta_spider.use_layer_pos_embeddings", 1);
    const int HD = ED / NH, FFN = ED * FFNX;
    fprintf(stderr, "HID=%d NL=%d ED=%d NB=%d NH=%d HD=%d FFN=%d pos=%d\n",
            HID, NL, ED, NB, NH, HD, FFN, USE_POS);

    std::vector<float> acts((size_t) NL * HID);
    FILE * fa = fopen(acts_path, "rb");
    if (!fa || fread(acts.data(), sizeof(float), acts.size(), fa) != acts.size()) {
        fprintf(stderr, "failed to read %s\n", acts_path); return 1; }
    fclose(fa);

    size_t mem = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    struct ggml_init_params cp = { mem, nullptr, /*no_alloc*/ true };
    struct ggml_context * c = ggml_init(cp);

    ggml_tensor * in_acts = ggml_new_tensor_2d(c, GGML_TYPE_F32, HID, NL);   // [HID, NL]
    ggml_set_name(in_acts, "in_acts"); ggml_set_input(in_acts);

    // per-layer projectors → x[ED, NL]
    ggml_tensor * x = nullptr;
    for (int i = 0; i < NL; ++i) {
        ggml_tensor * a = ggml_cont(c, ggml_view_2d(c, in_acts, HID, 1, in_acts->nb[1],
                                                    (size_t) i * in_acts->nb[1]));
        ggml_tensor * p = ggml_mul_mat(c, ggml_cont(c, Wf("enc.layer_projectors.%d.weight", i)), a); // [ED,1]
        x = x ? ggml_concat(c, x, p, 1) : p;                                  // → [ED, NL]
    }
    if (USE_POS) x = ggml_add(c, x, W("enc.layer_pos_embed"));                 // [ED,NL] + [ED,NL]
    x = ln(c, x, W("enc.input_norm.weight"), W("enc.input_norm.bias"));

    for (int b = 0; b < NB; ++b) {
        // --- self-attention ---
        ggml_tensor * res = x;
        ggml_tensor * h = ln(c, x, Wf("enc.blocks.%d.norm_attn.weight", b),
                                   Wf("enc.blocks.%d.norm_attn.bias", b));
        ggml_tensor * Q = ggml_mul_mat(c, ggml_cont(c, Wf("enc.blocks.%d.q_proj.weight", b)), h); // [ED,NL]
        ggml_tensor * K = ggml_mul_mat(c, ggml_cont(c, Wf("enc.blocks.%d.k_proj.weight", b)), h);
        ggml_tensor * V = ggml_mul_mat(c, ggml_cont(c, Wf("enc.blocks.%d.v_proj.weight", b)), h);
        Q = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, Q, HD, NH, NL), 0, 2, 1, 3)); // [HD,NL,NH]
        K = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, K, HD, NH, NL), 0, 2, 1, 3)); // [HD,NL,NH]
        ggml_tensor * kq = ggml_mul_mat(c, K, Q);                             // [NL(k), NL(q), NH]
        kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / sqrtf((float) HD), 0.0f);
        ggml_tensor * Vp = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, V, HD, NH, NL), 1, 2, 0, 3)); // [NL,HD,NH]
        ggml_tensor * kqv = ggml_mul_mat(c, Vp, kq);                          // [HD, NL(q), NH]
        kqv = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3));                 // [HD, NH, NL]
        ggml_tensor * att = ggml_reshape_2d(c, kqv, ED, NL);                  // [ED, NL]
        att = ggml_mul_mat(c, ggml_cont(c, Wf("enc.blocks.%d.o_proj.weight", b)), att);
        x = ggml_add(c, res, att);
        // --- FFN ---
        res = x;
        h = ln(c, x, Wf("enc.blocks.%d.norm_ffn.weight", b), Wf("enc.blocks.%d.norm_ffn.bias", b));
        h = ggml_mul_mat(c, ggml_cont(c, Wf("enc.blocks.%d.ffn.0.weight", b)), h);   // [FFN,NL]
        h = ggml_gelu_erf(c, h);
        h = ggml_mul_mat(c, ggml_cont(c, Wf("enc.blocks.%d.ffn.2.weight", b)), h);   // [ED,NL]
        x = ggml_add(c, res, h);
    }

    // output_proj + output_norm → [HID, NL]
    x = ln(c, x, W("enc.output_proj.0.weight"), W("enc.output_proj.0.bias"));
    x = ggml_mul_mat(c, ggml_cont(c, W("enc.output_proj.1.weight")), x);      // [HID, NL]
    x = ggml_gelu_erf(c, x);
    x = ggml_mul_mat(c, ggml_cont(c, W("enc.output_proj.3.weight")), x);      // [HID, NL]
    x = ln(c, x, W("enc.output_norm.weight"), W("enc.output_norm.bias"));
    ggml_set_name(x, "cog"); ggml_set_output(x);

    struct ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, x);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "graph alloc fail\n"); return 1; }
    ggml_backend_tensor_set(in_acts, acts.data(), 0, acts.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute fail\n"); return 1; }

    std::vector<float> cog((size_t) NL * HID);
    ggml_backend_tensor_get(x, cog.data(), 0, cog.size() * sizeof(float));
    FILE * fo = fopen(out_path, "wb"); fwrite(cog.data(), sizeof(float), cog.size(), fo); fclose(fo);
    fprintf(stderr, "→ %s (cog %d×%d)\n", out_path, NL, HID);

    ggml_gallocr_free(galloc); ggml_backend_free(backend);
    ggml_free(c); gguf_free(gguf); ggml_free(WCTX);
    return 0;
}
