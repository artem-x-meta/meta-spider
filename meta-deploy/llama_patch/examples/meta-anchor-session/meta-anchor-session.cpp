// meta-anchor-session — МНОГОХОДОВАЯ агентная сессия с GoalAnchor в llama.cpp.
//
// Цель якоря кодируется ОДИН раз в начале сессии (Pass-1 на СЫРОМ тексте цели, как
// GoalAnchor.set_anchor → tok(goal_text)); статичный cog держится и инъектится КАЖДЫЙ шаг
// на ВСЕХ ходах — обвязка «напоминает» модели цель, пока диалог растёт (тул-ответы, фидбэк,
// приманки). История каждый ход рендерится ВСТРОЕННЫМ chat-шаблоном модели (Qwen/Gemma/…),
// KV чистится и перекодируется (с инъекцией) — cog НЕ пересчитывается.
//
// Два режима:
//   ОДИНОЧНЫЙ — META_USER (+ META_OBS) → печать ходов в stdout (демо/A-B глазами).
//   БАТЧ (харнесс) — META_SESSIONS=<jsonl>: каждая строка {id, goal, system, user, obs:[...]}.
//     Модель грузится ОДИН раз; на КАЖДУЮ сессию якорь пере-энкодится из её goal; вывод —
//     META_SESS_OUT=<jsonl> строк {id, arm, turns:[code0,code1,...]}. Python-обёртка генерит
//     спеки/дрейф-сессии и AST-грейдит adherence (движок здесь, замерялка — снаружи).
//
// Env:
//   META_SIDECAR  — sidecar.gguf (CA + enc; kind=goal_anchor)
//   META_ANCHOR   — ТЕКСТ ЦЕЛИ (СЫРЬЁМ) для ОДИНОЧНОГО режима. Задаёт anchor-режим.
//   META_LAYERS   — csv target-слоёв (напр. 32,33,...,47)
//   META_SYSTEM / META_USER / META_OBS — одиночная сессия (см. выше)
//   META_SESSIONS / META_SESS_OUT — батч-режим (jsonl вход/выход)
//   META_NGEN     — макс токенов на ход (default 256)
//   META_BASE     — 1 = чистая база БЕЗ якоря (для A/B; в батче — арм "base")
// Модель/потоки — стандартные флаги (-m, -t, -c).

#include "arg.h"
#include "common.h"
#include "chat.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

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
    const char * sessions_file = getenv("META_SESSIONS");   // батч-режим
    const char * sess_out = getenv("META_SESS_OUT");
    const char * anchor_env = getenv("META_ANCHOR");
    const char * user0 = getenv("META_USER");
    const char * sys_env = getenv("META_SYSTEM");
    const char * obs_file = getenv("META_OBS");
    if (!sidecar || !env_lay || (!sessions_file && !user0)) {
        fprintf(stderr, "set META_SIDECAR / META_LAYERS / (META_USER | META_SESSIONS)\n");
        return 1;
    }
    const int n_gen = getenv("META_NGEN") ? atoi(getenv("META_NGEN")) : 256;
    const bool base_mode = getenv("META_BASE") && atoi(getenv("META_BASE")) != 0;

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

    if (!base_mode && llama_set_meta_adapter(ctx, sidecar) != 0) { fprintf(stderr, "meta adapter fail\n"); return 1; }
    auto tmpls = common_chat_templates_init(model, "");

    // энкод якоря из СЫРОГО текста цели (Pass-1) — держится до следующего encode
    auto encode_anchor = [&](const std::string & goal) {
        std::vector<llama_token> atok = common_tokenize(ctx, goal, true, true);
        std::vector<float> acts(layers.size() * cb.n_embd, 0.0f);
        cb.cur_out = acts.data();
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_decode(ctx, llama_batch_get_one(atok.data(), (int) atok.size()));
        cb.cur_out = nullptr;
        llama_meta_encode(ctx, acts.data(), (int) layers.size(), cb.n_embd);
    };

    // прогон одной сессии: system + user + [obs...] → код ассистента по ходам (cog держится).
    // Полный ре-декод каждый ход (KV чистится) — корректно и детерминированно; генерация на CPU
    // доминирует над префиллом, так что переиспользование KV не даёт выигрыша (проверено).
    auto run_turns = [&](const std::string & system, const std::string & user,
                         const std::vector<std::string> & observations) -> std::vector<std::string> {
        std::vector<common_chat_msg> msgs;
        if (!system.empty()) { common_chat_msg m; m.role = "system"; m.content = system; msgs.push_back(m); }
        { common_chat_msg m; m.role = "user"; m.content = user; msgs.push_back(m); }
        std::vector<std::string> turns;
        const size_t max_turns = observations.size() + 1;
        for (size_t turn = 0; turn < max_turns; ++turn) {
            common_chat_templates_inputs in;
            in.messages = msgs; in.add_generation_prompt = true; in.use_jinja = true; in.enable_thinking = false;
            std::string prompt = common_chat_templates_apply(tmpls.get(), in).prompt;
            std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
            llama_memory_clear(llama_get_memory(ctx), true);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int) tokens.size()))) break;
            std::string resp;
            for (int n = 0; n < n_gen; ++n) {
                float * logits = llama_get_logits_ith(ctx, -1);
                llama_token best = 0; float bv = logits[0];
                for (int i = 1; i < n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
                if (llama_vocab_is_eog(vocab, best)) break;
                resp += common_token_to_piece(ctx, best);
                // ранняя остановка: грейдим только ПЕРВЫЙ код-блок — как закрылся (2-й ```), стоп.
                // Режет длинные пояснения-хвосты (чистые потерянные токены на CPU).
                { size_t c = 0, p = 0; while ((p = resp.find("```", p)) != std::string::npos) { ++c; p += 3; }
                  if (c >= 2) break; }
                if (llama_decode(ctx, llama_batch_get_one(&best, 1))) break;
            }
            turns.push_back(resp);
            { common_chat_msg m; m.role = "assistant"; m.content = resp; msgs.push_back(m); }
            if (turn < observations.size()) { common_chat_msg m; m.role = "user"; m.content = observations[turn]; msgs.push_back(m); }
        }
        return turns;
    };

    // ───────────────────────── БАТЧ (харнесс) ─────────────────────────
    if (sessions_file) {
        std::ifstream f(sessions_file);
        std::ofstream out;
        if (sess_out) out.open(sess_out, std::ios::binary);
        std::string line; size_t idx = 0;
        const std::string arm = base_mode ? "base" : "anchor";
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            json s = json::parse(line, nullptr, false);
            if (s.is_discarded()) { fprintf(stderr, "bad json line %zu\n", idx); continue; }
            std::string id = s.value("id", std::to_string(idx));
            std::string goal = s.value("goal", "");
            std::string system = s.value("system", "");
            std::string user = s.value("user", "");
            std::vector<std::string> obs;
            if (s.contains("obs")) for (auto & o : s["obs"]) obs.push_back(o.get<std::string>());
            if (!base_mode && !goal.empty()) encode_anchor(goal);   // якорь под ЭТОТ спек
            std::vector<std::string> turns = run_turns(system, user, obs);
            json rec; rec["id"] = id; rec["arm"] = arm; rec["turns"] = turns;
            std::string js = rec.dump();
            if (sess_out) { out << js << "\n"; out.flush(); } else printf("%s\n", js.c_str());
            fprintf(stderr, "[harness] %s session %zu (%s) — %zu turns\n", arm.c_str(), idx, id.c_str(), turns.size());
            ++idx;
        }
        llama_backend_free();
        return 0;
    }

    // ───────────────────────── ОДИНОЧНЫЙ (демо) ─────────────────────────
    const bool anchor_mode = anchor_env && !base_mode;
    if (anchor_mode) { encode_anchor(std::string(anchor_env));
        fprintf(stderr, "[meta] anchor cog encoded from goal — held across all turns\n"); }
    std::vector<std::string> observations = obs_file ? read_split(obs_file) : std::vector<std::string>{};
    std::vector<std::string> turns = run_turns(sys_env ? sys_env : "", user0, observations);
    for (size_t t = 0; t < turns.size(); ++t)
        printf("\n=== TURN %zu (assistant%s) ===\n%s\n", t,
               anchor_mode ? "+anchor" : (base_mode ? " base" : ""), turns[t].c_str());
    llama_backend_free();
    return 0;
}
