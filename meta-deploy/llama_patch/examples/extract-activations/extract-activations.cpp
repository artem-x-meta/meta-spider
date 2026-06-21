// extract-activations — Milestone 0 (meta-spider llama.cpp deploy)
//
// Loads the GGUF model once, runs a set of prompts and dumps the residual stream
// (`l_out-{il}`) at the target layers for the LAST token of each prompt as raw f32.
//
// Args via env (to avoid fighting common_params_parse):
//   EXTRACT_LAYERS  — csv of layer indices, e.g. "10,14,18,22,25"
//   EXTRACT_PROMPTS — file with prompts separated by a \0 byte
//   EXTRACT_OUT     — output .bin: [n_prompts][n_layers][n_embd] f32 (C-order)
// Model/context — standard flags (-m, -c, -t).
//
// stderr output: "EXTRACT_META n_prompts=.. n_layers=.. n_embd=.. layers=.."

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct extract_cb_data {
    std::map<int, int> layer_to_slot;   // il -> slot index in the output buffer
    int                n_embd   = 0;
    float *            cur_out  = nullptr;  // current prompt's buffer [n_layers*n_embd]
    std::vector<uint8_t> scratch;
};

static float get_float(const uint8_t * data, ggml_type type, size_t off) {
    if (type == GGML_TYPE_F32) return *(const float *)(data + off);
    if (type == GGML_TYPE_F16) return ggml_fp16_to_fp32(*(const ggml_fp16_t *)(data + off));
    if (type == GGML_TYPE_BF16) return ggml_bf16_to_fp32(*(const ggml_bf16_t *)(data + off));
    return 0.0f;
}

// Parses "l_out-<int>" → layer number, or -1
static int parse_l_out(const char * name) {
    const char * pfx = "l_out-";
    size_t n = strlen(pfx);
    if (strncmp(name, pfx, n) != 0) return -1;
    const char * p = name + n;
    if (*p == '\0') return -1;
    for (const char * q = p; *q; ++q) if (*q < '0' || *q > '9') return -1;
    return atoi(p);
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (extract_cb_data *) user_data;
    if (ask) return true;
    if (d->cur_out == nullptr) return true;

    int il = parse_l_out(t->name);
    if (il < 0) return true;
    auto it = d->layer_to_slot.find(il);
    if (it == d->layer_to_slot.end()) return true;

    // t: ne[0]=n_embd, ne[1]=n_tokens. Take the last token (i1 = n_tokens-1).
    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    if (n_embd != d->n_embd || n_tokens < 1) return true;

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        d->scratch.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, d->scratch.data(), 0, ggml_nbytes(t));
        data = d->scratch.data();
    }

    const int64_t i1 = n_tokens - 1;
    float * out = d->cur_out + (size_t) it->second * n_embd;
    for (int64_t i0 = 0; i0 < n_embd; ++i0) {
        size_t off = i1 * t->nb[1] + i0 * t->nb[0];
        out[i0] = get_float(data, t->type, off);
    }
    return true;
}

static std::vector<std::string> read_prompts(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string blob = ss.str();
    std::vector<std::string> out;
    std::string cur;
    for (char c : blob) {
        if (c == '\0') { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * env_layers  = getenv("EXTRACT_LAYERS");
    const char * env_prompts = getenv("EXTRACT_PROMPTS");
    const char * env_out     = getenv("EXTRACT_OUT");
    if (!env_layers || !env_prompts || !env_out) {
        fprintf(stderr, "set EXTRACT_LAYERS / EXTRACT_PROMPTS / EXTRACT_OUT\n");
        return 1;
    }

    std::vector<int> layers;
    {
        std::stringstream ss(env_layers);
        std::string item;
        while (std::getline(ss, item, ',')) if (!item.empty()) layers.push_back(atoi(item.c_str()));
    }

    extract_cb_data cb_data;
    for (size_t i = 0; i < layers.size(); ++i) cb_data.layer_to_slot[layers[i]] = (int) i;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) { fprintf(stderr, "init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // add_bos=false: prompts (chat-template) already contain a literal <bos>;
    // otherwise a double BOS (won't match the fp16 reference).
    const bool add_bos = false;
    cb_data.n_embd = llama_model_n_embd(model);

    auto prompts = read_prompts(env_prompts);

    // Resume: skip the first N (already extracted), max per run (works around cumulative crash)
    size_t skip = 0, maxrun = prompts.size();
    if (const char * s = getenv("EXTRACT_SKIP")) skip = (size_t) atoll(s);
    if (const char * s = getenv("EXTRACT_MAX"))  maxrun = (size_t) atoll(s);

    fprintf(stderr, "EXTRACT_META n_prompts=%zu n_layers=%zu n_embd=%d layers=%s skip=%zu max=%zu\n",
            prompts.size(), layers.size(), cb_data.n_embd, env_layers, skip, maxrun);

    // Append mode for resume
    std::ofstream out(env_out, std::ios::binary | std::ios::app);
    std::vector<float> buf(layers.size() * cb_data.n_embd);

    size_t done = 0;
    for (size_t pi = skip; pi < prompts.size(); ++pi) {
        if (done >= maxrun) { fprintf(stderr, "max reached, stop at %zu\n", pi); break; }
        ++done;
        llama_memory_clear(llama_get_memory(ctx), true);
        std::fill(buf.begin(), buf.end(), 0.0f);
        cb_data.cur_out = buf.data();

        std::vector<llama_token> tokens = common_tokenize(ctx, prompts[pi], add_bos, true);
        if (tokens.empty()) { fprintf(stderr, "empty prompt %zu\n", pi); }
        else if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
            fprintf(stderr, "decode failed at prompt %zu\n", pi); return 1;
        }
        cb_data.cur_out = nullptr;
        out.write((const char *) buf.data(), buf.size() * sizeof(float));
        out.flush();  // don't lose progress on a crash
        if ((pi + 1) % 50 == 0) fprintf(stderr, "  %zu/%zu\n", pi + 1, prompts.size());
    }
    out.close();
    llama_backend_free();
    fprintf(stderr, "done -> %s\n", env_out);
    return 0;
}
