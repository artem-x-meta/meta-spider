// meta-anchor-session — МНОГОХОДОВАЯ агентная сессия с GoalAnchor в llama.cpp.
//
// Цель якоря кодируется ОДИН раз в начале сессии (Pass-1 на СЫРОМ тексте цели, как
// GoalAnchor.set_anchor → tok(goal_text)); статичный cog держится и инъектится КАЖДЫЙ шаг
// на ВСЕХ ходах — обвязка «напоминает» модели цель, пока диалог растёт (тул-ответы, фидбэк,
// приманки). История каждый ход рендерится ВСТРОЕННЫМ chat-шаблоном модели (Qwen/Gemma/…),
// KV чистится и перекодируется (с инъекцией) — cog НЕ пересчитывается.
//
// Это агентный аналог одноходового meta-generate: тот же lifecycle (encode once, hold),
// но с несколькими ходами ассистента, разделёнными обсервациями (тул/юзер).
//
// Env:
//   META_SIDECAR  — sidecar.gguf (CA + enc; kind=goal_anchor)
//   META_ANCHOR   — ТЕКСТ ЦЕЛИ (СЫРЬЁМ, без chat-обёртки). Задаёт anchor-режим.
//   META_LAYERS   — csv target-слоёв (напр. 32,33,...,47)
//   META_SYSTEM   — систем-промпт (опц.)
//   META_USER     — начальный ход пользователя (спек задачи)
//   META_OBS      — файл с обсервациями (\0-разделённые): вставляются как user-ход ПОСЛЕ
//                   каждого ответа ассистента. N обсерваций → до N+1 ходов ассистента.
//   META_NGEN     — макс токенов на ход (default 256)
//   META_BASE     — 1 = чистая база БЕЗ якоря (для A/B того же диалога)
//   META_GAIN     — сила инъекции (default 1.0)
// Модель/потоки — стандартные флаги (-m, -t, -c).

#include "arg.h"
#include "common.h"
#include "chat.h"
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

struct tap_cb {
    std::map<int,int> layer_to_slot;
    int n_embd = 0;
    float * cur_out = nullptr;       // буфер Pass-1 [n_layers*n_embd]; null вне Pass-1
    std::vector<uint8_t> scratch;
};

static float getf(const uint8_t * d, ggml_type t, size_t off) {
    if (t == GGML_TYPE_F32)  return *(const float *)(d+off);
    if (t == GGML_TYPE_F16)  return ggml_fp16_to_fp32(*(const ggml_fp16_t *)(d+off));
    if (t == GGML_TYPE_BF16) return ggml_bf16_to_fp32(*(const ggml_bf16_t *)(d+off));
    return 0.0f;
}
static int parse_l_out(const char * name) {
    const char * p = "l_out-"; size_t n = strlen(p);
    if (strncmp(name, p, n) != 0) return -1;
    const char * q = name + n; if (!*q) return -1;
    for (const char * r = q; *r; ++r) if (*r < '0' || *r > '9') return -1;
    return atoi(q);
}
static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * d = (tap_cb *) ud;
    if (ask) return true;
    if (!d->cur_out) return true;
    int il = parse_l_out(t->name);
    if (il < 0) return true;
    auto it = d->layer_to_slot.find(il);
    if (it == d->layer_to_slot.end()) return true;
    const int64_t ne = t->ne[0], nt = t->ne[1];
    if (ne != d->n_embd || nt < 1) return true;
    const uint8_t * data;
    if (ggml_backend_buffer_is_host(t->buffer)) data = (const uint8_t *) t->data;
    else { d->scratch.resize(ggml_nbytes(t)); ggml_backend_tensor_get(t, d->scratch.data(), 0, ggml_nbytes(t)); data = d->scratch.data(); }
    float * out = d->cur_out + (size_t) it->second * ne;
    for (int64_t i0 = 0; i0 < ne; ++i0) out[i0] = getf(data, t->type, (nt-1)*t->nb[1] + i0*t->nb[0]);
    return true;
}

// \0-разделённый файл → список строк
static std::vector<std::string> read_split(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    std::string blob = ss.str(); std::vector<std::string> out; std::string cur;
    for (char c : blob) { if (c == '\0') { out.push_back(cur); cur.clear(); } else cur.push_back(c); }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    const char * sidecar = getenv("META_SIDECAR");
    const char * env_lay = getenv("META_LAYERS");
    const char * anchor_env = getenv("META_ANCHOR");
    const char * user0 = getenv("META_USER");
    const char * sys_env = getenv("META_SYSTEM");
    const char * obs_file = getenv("META_OBS");
    if (!sidecar || !env_lay || !user0) {
        fprintf(stderr, "set META_SIDECAR / META_LAYERS / META_USER (+ META_ANCHOR for the anchor)\n");
        return 1;
    }
    const int n_gen = getenv("META_NGEN") ? atoi(getenv("META_NGEN")) : 256;
    const bool base_mode = getenv("META_BASE") && atoi(getenv("META_BASE")) != 0;
    const bool anchor_mode = anchor_env && !base_mode;

    std::vector<int> layers;
    { std::stringstream ss(env_lay); std::string it; while (std::getline(ss, it, ',')) if (!it.empty()) layers.push_back(atoi(it.c_str())); }
    tap_cb cb;
    for (size_t i = 0; i < layers.size(); ++i) cb.layer_to_slot[layers[i]] = (int) i;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &cb;
    params.warmup = false;

    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) { fprintf(stderr, "init failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    cb.n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // meta-адаптер (кроме base-режима)
    if (!base_mode && llama_set_meta_adapter(ctx, sidecar) != 0) { fprintf(stderr, "meta adapter fail\n"); return 1; }

    // встроенный chat-шаблон модели (Qwen/Gemma/… из GGUF)
    auto tmpls = common_chat_templates_init(model, "");

    // ── якорь: Pass-1 на СЫРОМ тексте цели ОДИН раз → cog держится весь диалог ──
    if (anchor_mode) {
        std::vector<llama_token> atok = common_tokenize(ctx, std::string(anchor_env), true, true);
        std::vector<float> acts(layers.size() * cb.n_embd, 0.0f);
        cb.cur_out = acts.data();
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_decode(ctx, llama_batch_get_one(atok.data(), (int) atok.size()));
        cb.cur_out = nullptr;
        llama_meta_encode(ctx, acts.data(), (int) layers.size(), cb.n_embd);
        fprintf(stderr, "[meta] anchor cog encoded from goal (%zu tok) — held across all turns\n", atok.size());
    }

    std::vector<std::string> observations = obs_file ? read_split(obs_file) : std::vector<std::string>{};

    // история сессии
    std::vector<common_chat_msg> msgs;
    if (sys_env && *sys_env) { common_chat_msg m; m.role = "system"; m.content = sys_env; msgs.push_back(m); }
    { common_chat_msg m; m.role = "user"; m.content = user0; msgs.push_back(m); }

    auto render = [&]() -> std::string {
        common_chat_templates_inputs in;
        in.messages = msgs;
        in.add_generation_prompt = true;
        in.use_jinja = true;
        in.enable_thinking = false;
        return common_chat_templates_apply(tmpls.get(), in).prompt;
    };

    const size_t max_turns = observations.size() + 1;
    for (size_t turn = 0; turn < max_turns; ++turn) {
        std::string prompt = render();
        std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int) tokens.size()))) {
            fprintf(stderr, "decode fail (turn %zu)\n", turn); break;
        }
        std::string resp;
        for (int n = 0; n < n_gen; ++n) {
            float * logits = llama_get_logits_ith(ctx, -1);
            llama_token best = 0; float bv = logits[0];
            for (int i = 1; i < n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
            if (llama_vocab_is_eog(vocab, best)) break;
            resp += common_token_to_piece(ctx, best);
            if (llama_decode(ctx, llama_batch_get_one(&best, 1))) break;
        }
        printf("\n=== TURN %zu (assistant%s) ===\n%s\n", turn, anchor_mode ? "+anchor" : (base_mode ? " base" : ""), resp.c_str());
        fflush(stdout);
        { common_chat_msg m; m.role = "assistant"; m.content = resp; msgs.push_back(m); }
        if (turn < observations.size()) {
            common_chat_msg m; m.role = "user"; m.content = observations[turn]; msgs.push_back(m);
        }
    }
    llama_backend_free();
    return 0;
}
