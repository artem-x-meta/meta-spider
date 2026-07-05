// meta-generate — двухпроходный инференс Скептика (meta-spider) в llama.cpp.
//
// Pass 1 (чистый): прогон промпта, cb_eval тапит residual l_out-{il} target-слоёв
//   (последний токен). cog ещё не задан → meta-адаптер НЕ инъектит (Pass-1 чистый).
// Между: активации → meta_encoder.exe (внешний ggml-энкодер) → cognitive tokens.
// Pass 2 (инъекция): llama_set_meta_cog включает CA-инъекцию; чистим KV, декодим
//   промпт заново (теперь с инъекцией) и greedy-генерируем ответ.
//
// Anchor-режим (GoalAnchor): META_ANCHOR="<текст цели/спека>" → cog кодируется из активаций
//   ЦЕЛИ ОДИН раз (Pass-1 на тексте цели, до цикла) и держится статично на ВСЕЙ генерации
//   (trigger=always — валидированный режим). Тогда META_PROMPT/META_PROMPTS — рабочий контекст,
//   а не источник cog; per-prompt Pass-1 и dynamic-рефреш выключаются. Без META_ANCHOR — режим
//   Doubter (cog из самого промпта, как раньше).
//
// Env:
//   META_SIDECAR  — sidecar.gguf (CA + enc веса; kind=doubter|goal_anchor)
//   META_ANCHOR   — текст цели для якоря (включает anchor-режим; cog из него, один раз)
//   META_LAYERS   — csv target-слоёв (напр. 10,14,18,22,25)
//   META_PROMPT   — текст вопроса/контекст (без chat-обёртки; обернём в Gemma-формат)
//   META_NGEN     — макс токенов генерации (default 200)
//   META_TMP      — каталог для acts.bin/cog.bin (default .)
// Модель/потоки — стандартные флаги (-m, -t, -c).

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
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
    const char * p = "l_out-";
    size_t n = strlen(p);
    if (strncmp(name, p, n) != 0) return -1;
    const char * q = name + n;
    if (!*q) return -1;
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

static std::vector<std::string> read_prompts(const std::string & path) {
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
    const char * prompt  = getenv("META_PROMPT");
    const char * prompts_file = getenv("META_PROMPTS");   // batch: \0-разделённые
    const char * out_file = getenv("META_OUT");           // batch output (\0-разделённый)
    if (!sidecar || !env_lay || (!prompt && !prompts_file)) {
        fprintf(stderr, "set META_SIDECAR / META_LAYERS / (META_PROMPT | META_PROMPTS)\n"); return 1;
    }
    const int n_gen = getenv("META_NGEN") ? atoi(getenv("META_NGEN")) : 200;
    std::string tmp = getenv("META_TMP") ? getenv("META_TMP") : ".";

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

    const bool base_mode = getenv("META_BASE") && atoi(getenv("META_BASE")) != 0;
    const bool dynamic = getenv("META_DYNAMIC") && atoi(getenv("META_DYNAMIC")) != 0;
    const float thr = getenv("META_THRESHOLD") ? atof(getenv("META_THRESHOLD")) : 0.5f;
    const int min_iv = 3, max_iv = 20;
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // base-режим = без meta-адаптера (чистая база, оракул); иначе ставим адаптер
    if (!base_mode && llama_set_meta_adapter(ctx, sidecar) != 0) { fprintf(stderr, "meta adapter fail\n"); return 1; }

    auto cossim = [&](const std::vector<float>&a, const std::vector<float>&b){
        double dot=0,na=0,nb=0; for (size_t i=0;i<a.size();++i){dot+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];}
        return (na>0&&nb>0)? dot/(sqrt(na)*sqrt(nb)) : 1.0; };

    const bool raw_prompt = getenv("META_RAW") && atoi(getenv("META_RAW")) != 0;
    const char * anchor_env = getenv("META_ANCHOR");
    const bool anchor_mode = anchor_env && !base_mode;   // GoalAnchor: cog из цели, один раз, статично
    auto wrap = [&](const std::string & s) {
        return raw_prompt ? s
            : std::string("<start_of_turn>user\n") + s + "<end_of_turn>\n<start_of_turn>model\n"; };

    // Anchor: Pass-1 на ТЕКСТЕ ЦЕЛИ один раз → cog держится в адаптере на все промпты (persist).
    // ВАЖНО: цель токенизируется СЫРЬЁМ (как GoalAnchor.set_anchor → tok(goal_text): BOS + токены,
    // БЕЗ chat-шаблона). Оборачивать её в <|im_start|> нельзя — энкодер учился на сырых активациях
    // структурированного spec_text ("TASK:\n\nREQUIREMENTS (all mandatory):\n1. ..."). wrap() — только промпту.
    if (anchor_mode) {
        std::vector<llama_token> atok = common_tokenize(ctx, std::string(anchor_env), true, true);
        std::vector<float> acts(layers.size() * cb.n_embd, 0.0f);
        cb.cur_out = acts.data();
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_decode(ctx, llama_batch_get_one(atok.data(), (int) atok.size()));
        cb.cur_out = nullptr;
        llama_meta_encode(ctx, acts.data(), (int) layers.size(), cb.n_embd);
        fprintf(stderr, "[meta] anchor cog encoded from goal (%zu tok)\n", atok.size());
    }

    auto run_one = [&](const std::string & q) -> std::string {
        // META_RAW=1 → промпт как есть (для не-Gemma моделей передай свой chat-формат в META_PROMPT)
        std::string text = wrap(q);
        std::vector<llama_token> tokens = common_tokenize(ctx, text, true, true);
        std::vector<float> acts(layers.size() * cb.n_embd, 0.0f);
        if (!base_mode && !anchor_mode) {
            // Doubter Pass-1: чистый тап (cog ещё нули → инъекции нет) → энкодер, per-prompt
            cb.cur_out = acts.data();
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
            cb.cur_out = nullptr;
            llama_meta_encode(ctx, acts.data(), (int) layers.size(), cb.n_embd);
        }
        // Pass-2 (инъекция) или единственный проход (base)
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
        std::vector<float> cached = acts, step(acts.size(), 0.0f);
        std::string outtext; int since = 0, cur_iv = min_iv;
        const bool do_dynamic = dynamic && !base_mode && !anchor_mode;   // якорь статичен → без рефреша
        for (int n = 0; n < n_gen; ++n) {
            float * logits = llama_get_logits_ith(ctx, -1);
            llama_token best = 0; float bv = logits[0];
            for (int i = 1; i < n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
            if (llama_vocab_is_eog(vocab, best)) break;
            outtext += common_token_to_piece(ctx, best);
            if (do_dynamic) cb.cur_out = step.data();
            if (llama_decode(ctx, llama_batch_get_one(&best, 1))) break;
            cb.cur_out = nullptr;
            if (do_dynamic) {
                if (++since >= cur_iv) {
                    if (since >= max_iv || cossim(step, cached) < thr) {
                        llama_meta_encode(ctx, step.data(), (int) layers.size(), cb.n_embd);
                        cached = step; since = 0; cur_iv = min_iv;
                    } else { cur_iv = std::min(cur_iv + min_iv, max_iv); since = 0; }
                }
            }
        }
        return outtext;
    };

    std::vector<std::string> prompts = prompts_file ? read_prompts(prompts_file)
                                                    : std::vector<std::string>{prompt};
    std::ofstream out;
    if (out_file) out.open(out_file, std::ios::binary);
    for (size_t i = 0; i < prompts.size(); ++i) {
        fprintf(stderr, "[meta] prompt %zu/%zu start\n", i, prompts.size()); fflush(stderr);
        std::string r = run_one(prompts[i]);
        for (char & ch : r) if (ch == '\n' || ch == '\r') ch = ' ';   // одна строка/промпт
        if (out_file) { out.write(r.data(), r.size()); out.put('\0'); out.flush(); }
        else printf("\n=== META OUTPUT ===\n%s\n", r.c_str());
        if (prompts_file && (i + 1) % 10 == 0) fprintf(stderr, "  %zu/%zu\n", i + 1, prompts.size());
    }
    llama_backend_free();
    return 0;
}
