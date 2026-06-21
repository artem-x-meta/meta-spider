// M3a: ggml forward of BottleneckCrossAttention (standalone validation against PyTorch).
// out = hidden + tanh(gate)·up( MHA(q=q_proj(down(norm(hidden))), k=k_proj(cog), v=v_proj(cog))
//                               + token_preference_bias )
// 8 heads×32, scale 1/√32, token_preference — additive bias on the scores before softmax.

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

// Sizes — generic (phase 4a): from the shape of the CA tensors + sidecar metadata (not hardcoded).
static const float EPS = 1e-5f;

static struct ggml_context * WCTX = nullptr;

static ggml_tensor * W(const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(WCTX, name.c_str());
    if (!t) { fprintf(stderr, "MISSING tensor: %s\n", name.c_str()); exit(1); }
    return t;
}
static ggml_tensor * ln(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_add(c, ggml_mul(c, ggml_norm(c, x, EPS), w), b);
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s sidecar.gguf layer hidden.bin cog.bin out.bin [seq]\n", argv[0]);
        return 1;
    }
    const char * sidecar = argv[1];
    std::string L = argv[2];
    const char * hid_path = argv[3];
    const char * cog_path = argv[4];
    const char * out_path = argv[5];
    int seq = argc > 6 ? atoi(argv[6]) : 3;
    std::string pfx = "ca." + L + ".";

    struct gguf_init_params gp = { false, &WCTX };
    struct gguf_context * gguf = gguf_init_from_file(sidecar, gp);
    if (!gguf) { fprintf(stderr, "failed to open %s\n", sidecar); return 1; }

    // sizes from the shape of the CA tensors + metadata (generic, not hardcoded)
    ggml_tensor * dpw = W(pfx + "down_proj.weight");          // ne=[HID, BN]
    const int HID  = (int) dpw->ne[0];
    const int BN   = (int) dpw->ne[1];
    const int NCOG = (int) W(pfx + "token_preference")->ne[0];
    int64_t k_nh = gguf_find_key(gguf, "meta_spider.ca_num_heads");
    const int NHEAD = k_nh >= 0 ? (int) gguf_get_val_u32(gguf, k_nh) : 8;
    const int HD = BN / NHEAD;
    fprintf(stderr, "CA[%s]: HID=%d BN=%d NCOG=%d NHEAD=%d\n", L.c_str(), HID, BN, NCOG, NHEAD);

    std::vector<float> hid(seq * HID), cog(NCOG * HID);
    FILE * f = fopen(hid_path, "rb"); fread(hid.data(), 4, hid.size(), f); fclose(f);
    f = fopen(cog_path, "rb"); fread(cog.data(), 4, cog.size(), f); fclose(f);

    // tanh(gate) — scalar from the loaded tensor
    float gate_raw = ((float *) W(pfx + "gate")->data)[0];
    float gate = tanhf(gate_raw);

    size_t mem = ggml_tensor_overhead() * 1024 + ggml_graph_overhead();
    struct ggml_init_params cp = { mem, nullptr, true };
    struct ggml_context * c = ggml_init(cp);

    ggml_tensor * in_h = ggml_new_tensor_2d(c, GGML_TYPE_F32, HID, seq);
    ggml_set_name(in_h, "hidden"); ggml_set_input(in_h);
    ggml_tensor * in_c = ggml_new_tensor_2d(c, GGML_TYPE_F32, HID, NCOG);
    ggml_set_name(in_c, "cog"); ggml_set_input(in_c);

    ggml_tensor * h = ln(c, in_h, W(pfx + "norm.weight"), W(pfx + "norm.bias"));
    ggml_tensor * hc = ggml_mul_mat(c, W(pfx + "down_proj.weight"), h);   // [BN, seq]
    ggml_tensor * Q = ggml_mul_mat(c, W(pfx + "q_proj.weight"), hc);      // [BN, seq]
    ggml_tensor * K = ggml_mul_mat(c, W(pfx + "k_proj.weight"), in_c);    // [BN, NCOG]
    ggml_tensor * V = ggml_mul_mat(c, W(pfx + "v_proj.weight"), in_c);    // [BN, NCOG]

    Q = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, Q, HD, NHEAD, seq),  0, 2, 1, 3)); // [HD,seq,NHEAD]
    K = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, K, HD, NHEAD, NCOG), 0, 2, 1, 3)); // [HD,NCOG,NHEAD]
    ggml_tensor * kq = ggml_mul_mat(c, K, Q);                       // [NCOG, seq, NHEAD]
    kq = ggml_scale(c, kq, 1.0f / sqrtf((float) HD));
    // token_preference [NCOG] broadcast over seq+heads → add before softmax
    kq = ggml_add(c, kq, W(pfx + "token_preference"));
    kq = ggml_soft_max(c, kq);
    ggml_tensor * Vp = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, V, HD, NHEAD, NCOG), 1, 2, 0, 3)); // [NCOG,HD,NHEAD]
    ggml_tensor * kqv = ggml_mul_mat(c, Vp, kq);                    // [HD, seq, NHEAD]
    kqv = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3));          // [HD, NHEAD, seq]
    ggml_tensor * att = ggml_reshape_2d(c, kqv, BN, seq);          // [BN, seq]
    ggml_tensor * out = ggml_mul_mat(c, W(pfx + "up_proj.weight"), att);  // [HID, seq]
    out = ggml_add(c, in_h, ggml_scale(c, out, gate));            // residual + tanh(gate)*out
    ggml_set_output(out);

    struct ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);
    ggml_backend_tensor_set(in_h, hid.data(), 0, hid.size() * 4);
    ggml_backend_tensor_set(in_c, cog.data(), 0, cog.size() * 4);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute fail\n"); return 1;
    }

    std::vector<float> res(seq * HID);
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * 4);
    f = fopen(out_path, "wb"); fwrite(res.data(), 4, res.size(), f); fclose(f);
    fprintf(stderr, "→ %s  (out %d×%d, gate=%.4f)\n", out_path, seq, HID, gate);

    ggml_gallocr_free(galloc); ggml_backend_free(backend);
    ggml_free(c); gguf_free(gguf); ggml_free(WCTX);
    return 0;
}
