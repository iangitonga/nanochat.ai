#include "gten/gten.h"
#include "tinyllama.h"
#include "tinyllama_tok.h"
#include "zephyr.h"
#include "zephyr_tok.h"

#include <random>
#include <algorithm>
#include <functional>


template <typename ModelT, typename TokenizerT>
struct InferencePackage {
    ModelT* model_ptr;
    TokenizerT* tokenizer_ptr;
    std::string model_name;

    InferencePackage(ModelT* mptr, TokenizerT* tptr, const std::string& model_name_)
        : model_ptr{mptr}, tokenizer_ptr{tptr}, model_name{model_name_}
    {}
};


void* init_inference_package(const std::string& model_name, Dtype model_dtype, const std::string& model_path, const std::string& tokenizer_path, int n_ctx)
{
    std::cout << "Loading package ...\n";

    ModuleDtype dtype;
    if (model_dtype == kFloat16) {
        dtype = { .wdtype=kFloat16, .adtype=kFloat16 };
    } else if (model_dtype == kQint8) {
        dtype = { .wdtype=kQint8, .adtype=kQint8 };
    } else {
        dtype = { .wdtype=kQint4, .adtype=kQint8 };
    }

    void* infpkg_ptr;
    if (model_name == "tinyllama") {
        TinyLlama* model_ptr = new TinyLlama{dtype, n_ctx};
        std::ifstream fin{model_path, std::ios_base::binary};
        if (!fin.is_open()) { std::cout << "Fin failed ...\n";return nullptr; }
        model_ptr->load_from_ckpt(fin);

        TinyLlamaTokenizer* tok_ptr = new TinyLlamaTokenizer{tokenizer_path.c_str(), 32000};

        infpkg_ptr = new InferencePackage{model_ptr, tok_ptr, model_name};
    } else {
        Zephyr1_6b* model_ptr = new Zephyr1_6b{n_ctx, dtype};
        std::ifstream fin{model_path, std::ios_base::binary};
        if (!fin.is_open()) { std::cout << "Fin failed ...\n";return nullptr; }
        model_ptr->load_from_ckpt(fin);

        void* tok_ptr = new ZephyrTokenizer{tokenizer_path, 100352};
        infpkg_ptr = new InferencePackage{model_ptr, tok_ptr, model_name};
    }

    std::cout << "Loading package complete!\n";

    return infpkg_ptr;
}


template <typename ModelT, typename TokenizerT>
void perform_inference(void* pkg_ptr, std::string& prompt, std::function<void(const char*)> callback_function)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    InferencePackage<ModelT, TokenizerT>* pkg = reinterpret_cast<InferencePackage<ModelT, TokenizerT>*>(pkg_ptr);

    std::vector<int> tokens = pkg->tokenizer_ptr->encode(prompt);
    const int n_predict = pkg->model_ptr->n_ctx_;
    tokens.reserve(n_predict);

    const int logits_size = pkg->model_ptr->params.n_vocab;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);

    const float temp = 0.9f;
    const int top_k = 50;
    const int eot_token = pkg->tokenizer_ptr->eos;
    const int max_iters = n_predict - tokens.size();
    int n_iters = 0;
    bool reached_eot = false;
    for (int i = 0; i < max_iters; i++)
    {
        n_iters += 1;

        Tensor input{tokens.data(), {(int)tokens.size()}, kInt32};

        const int start_pos = (i == 0) ? 0 : input.numel() - 1; 
        Tensor logits = pkg->model_ptr->logits(input, start_pos);

        const float* logits_data = logits.data_ptr<float>();

        logits_probs.clear();
        for (int j = 0; j < logits_size; ++j) {
            logits_probs.push_back(std::make_pair((double)logits_data[j] / temp, j));
        }
        
        // Select top k elements.
        std::partial_sort(
                logits_probs.begin(),
                logits_probs.begin() + top_k,
                logits_probs.end(),
                [](const std::pair<double, int> &rhs, const std::pair<double, int> &lhs) {
            return rhs.first > lhs.first;
        });
        logits_probs.resize(top_k);
        
        // compute softmax
        double sum_exp = 0;
        for (int j = 0; j < top_k; ++j)
        {
            logits_probs[j].first = std::exp(logits_probs[j].first);
            sum_exp += logits_probs[j].first;
        }
        for (int j = 0; j < top_k; ++j)
            logits_probs[j].first = logits_probs[j].first / sum_exp;

        std::vector<double> probs(logits_size, 0.0);
        for (int j = 0; j < top_k; j++)
        {
            const auto &prob_pair = logits_probs[j];
            probs[prob_pair.second] = prob_pair.first;
        }

        std::discrete_distribution dist(probs.begin(), probs.end());
        uint32_t pred_token = dist(gen);

        // if (int(pred_token) == eot_token || (pred_token >= 130 && pred_token < 259)) {
        if (int(pred_token) == eot_token) {
            // std::cout << "<EOT>\n";
            callback_function("<endoftext>");
            reached_eot = true;
            break;
        }
        const int prev_token = (i == 0) ? 1 : tokens.back();
        // std::cout << pred_token << " ";


        const char* piece = pkg->tokenizer_ptr->decode(prev_token, pred_token);
        {
            callback_function(piece);
            // std::cout << pred_token << ": " << piece << '\n';
            // // piece might be a raw byte token, and we only want to print printable chars or whitespace
            // // because some of the other bytes can be various control codes, backspace, etc.
            // if (piece == NULL) { std::cout << "NULL/0: " << pred_token << "\n"; }
            // else if (piece[0] == '\0') { std::cout << "NULL/1: " << pred_token << "\n"; }
            // else if (piece[1] == '\0') {
            //     unsigned char byte_val = piece[0];
            //     if (!(isprint(byte_val) || isspace(byte_val))) {
            //         std::cout << "BAD: " << pred_token << "\n"; // bad byte, don't print it
            //     }
            // }
            // else {
                
            // }
        }

        tokens.push_back(pred_token);
    }

    if (!reached_eot) {
        callback_function("<endoftext>");
    }
}
