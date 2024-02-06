 // <|user|>
// Which famous math number begins with 1.6 ...?<|endoftext|>
// <|assistant|>

#pragma once

#include <random>
#include <algorithm>

#include "gten/gten.h"


using namespace gten;


struct Zephyr1_6bParams {
    const int n_vocab = 100352;
    const int max_ctx = 4096;
    const int n_embd = 2048;
    const int n_ffn = 5632;
    const int n_layers = 24;
    const int n_heads = 32;
    const int n_query_groups = 32;
    const float rope_pct = 0.25f;
    const int eot_token = 100257;
};


class Zephyr1_6b {
public:
    const Zephyr1_6bParams params = Zephyr1_6bParams{};
    ModuleDtype dtype_;
    int n_ctx_;

public:
    Zephyr1_6b(const int n_ctx, ModuleDtype dtype)
        : dtype_{dtype},
          n_ctx_{n_ctx},
          tok_emb_{Embedding(params.n_vocab, params.n_embd, n_ctx, dtype)},
          norm_{LayerNorm(params.n_embd, n_ctx, {kFloat16, dtype.adtype})},
          lm_head_{EmbeddingLinear{params.n_embd, params.n_vocab, n_ctx, {dtype.wdtype, kFloat32}}}
    {
        blocks_.reserve(params.n_layers);
        for (int i = 0; i < params.n_layers; i++) {
            blocks_.push_back(
                AttentionBlock2(params.n_heads, params.n_embd, params.n_query_groups, params.n_ffn, n_ctx, dtype, params.rope_pct, /*qkv_bias=*/true)
            );
        }
    }

    Tensor logits(const Tensor& tokens, const int start_pos=0) {
        if (tokens.numel() > n_ctx_) {
            std::cerr << "Number of prompt tokens (" << tokens.numel() << ") exceed provided maximum ctx size (" << n_ctx_ << ")\n";
            std::exit(EXIT_FAILURE);
        }

        Tensor logits = tok_emb_.forward(tokens, start_pos);

        for (auto& block : blocks_) {
            logits = block.forward(logits, start_pos);
        }

        logits = norm_.forward(logits, start_pos);
        logits = lm_head_.forward(logits);

        return logits;
    }

    void load_from_ckpt(std::ifstream& ckpt);

private:
    Embedding tok_emb_;
    LayerNorm norm_;
    EmbeddingLinear lm_head_;
    std::vector<AttentionBlock2> blocks_;
};


// static inline void read_into_weight(
//     std::ifstream& fin, gten::Tensor& tensor, ModuleDtype dtype)
// {
//     std::string weight_name;
//     int32_t weight_name_size;
//     fin.read(reinterpret_cast<char*>(&weight_name_size), sizeof(weight_name_size));
//     weight_name.resize(weight_name_size);
//     fin.read(reinterpret_cast<char*>(weight_name.data()), weight_name_size);

//     int32_t weight_payload_size;
//     fin.read(reinterpret_cast<char*>(&weight_payload_size), sizeof(weight_payload_size));


//     GTEN_ASSERTM(
//         static_cast<size_t>(weight_payload_size) == tensor.nbytes(),
//         "Weight `%s` data size: %d does not match the expected size: %ld.",
//         weight_name.c_str(), weight_payload_size, tensor.nbytes());
//     fin.read(tensor.data_ptr<char>(), weight_payload_size);
// }


// static inline void read_layer_header(std::ifstream& fin, bool debug = false) {
//     std::string layer_name;
//     int32_t layer_name_size;
//     fin.read(reinterpret_cast<char*>(&layer_name_size), sizeof(layer_name_size));
//     layer_name.resize(layer_name_size);
//     fin.read(reinterpret_cast<char*>(layer_name.data()), layer_name_size);

//     if (debug) {
//         std::cout << "Layer: " << layer_name << "\n";
//     }
// }

void Zephyr1_6b::load_from_ckpt(std::ifstream &ckpt)
{
    const int64_t expected_magic = 0x454c49464e455447;
    int64_t magic;
    ckpt.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERTM(magic == expected_magic, "Magic number in the binary does not match the expected one.\n");

    read_layer_header(ckpt);
    read_into_weight(ckpt, tok_emb_.weight, dtype_);

    for (auto& block : blocks_)
    {
        // q_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.query.weight, dtype_);

        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.query.bias, {kFloat16, dtype_.adtype});

        // k_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.key.weight, dtype_);

        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.key.bias, {kFloat16, dtype_.adtype});

        // v_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.value.weight, dtype_);

        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.value.bias, {kFloat16, dtype_.adtype});

        // o_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.qkv_proj.weight, dtype_);

        // ffn_gate_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_gate_proj.weight, dtype_);

        // ffn_up_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_up_proj.weight, dtype_);

        // ffn_down_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_down_proj.weight, dtype_);

        // attn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn_norm.weight, {kFloat16, dtype_.adtype});
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn_norm.bias, {kFloat16, dtype_.adtype});

        // ffn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_norm.weight, {kFloat16, dtype_.adtype});
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_norm.bias, {kFloat16, dtype_.adtype});
    }
    
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.weight, {kFloat16, dtype_.adtype});
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.bias, {kFloat16, dtype_.adtype});

    read_layer_header(ckpt);
    read_into_weight(ckpt, lm_head_.weight, dtype_);
}
