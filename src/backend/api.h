#include "gten/gten.h"
#include "tinyllama/tinyllama.h"
#include "zephyr/zephyr.h"
#include "minicpm/minicpm.h"

#include <random>
#include <algorithm>
#include <functional>


struct InferencePackage {
    Model* model_ptr;
    Tokenizer* tokenizer_ptr;
    std::string model_name;

    InferencePackage(Model* mptr, Tokenizer* tptr, const std::string& model_name_)
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

    std::ifstream fin{model_path, std::ios_base::binary};
    if (!fin.is_open()) {
        // This should never happen because the frontend checks if the file exists and is readable.
        std::cout << "Unexpected error: path failed to open: " << model_path << "\n";
        return nullptr;
    }

    void* infpkg_ptr;
    if (model_name == "minicpm") {
        MiniCPM* model_ptr = new MiniCPM{n_ctx, dtype};
        model_ptr->load_from_ckpt(fin);

        const std::string prompt_prefix = "<用户>";
        const std::string prompt_suffix = "<AI>";
        LLamaTokenizer* tok_ptr = new LLamaTokenizer{tokenizer_path.c_str(), minicpm_cfg.n_vocab, minicpm_cfg.eos, prompt_prefix, prompt_suffix, {}, {}};

        infpkg_ptr = new InferencePackage{model_ptr, tok_ptr, model_name};
    }
    else if (model_name == "tinyllama") {
        TinyLLama* model_ptr = new TinyLLama{n_ctx, dtype};
        model_ptr->load_from_ckpt(fin);

        const int vocab_size = tinyllama_cfg.n_vocab - 3;
        const std::vector<int> prefix_tokens = {1, 32001};
        const std::vector<int> suffix_tokens = {32002, 29871, 13, 32001, 20255, 13};
        LLamaTokenizer* tok_ptr = new LLamaTokenizer{tokenizer_path.c_str(), vocab_size, tinyllama_cfg.eos, "user\n", "", prefix_tokens, suffix_tokens};

        infpkg_ptr = new InferencePackage{model_ptr, tok_ptr, model_name};
    } else {
        Zephyr* model_ptr = new Zephyr{n_ctx, dtype};
        model_ptr->load_from_ckpt(fin);

        Gpt2Tokenizer* tok_ptr = new Gpt2Tokenizer{tokenizer_path, zephyr_cfg.n_vocab, zephyr_cfg.eos};

        infpkg_ptr = new InferencePackage{model_ptr, tok_ptr, model_name};
    }

    std::cout << "Loading package complete!\n";

    return infpkg_ptr;
}


void perform_inference(void* pkg_ptr, std::string& prompt, std::function<void(const char*)> callback_function)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    InferencePackage* pkg = reinterpret_cast<InferencePackage*>(pkg_ptr);

    std::vector<int> tokens = pkg->tokenizer_ptr->encode(prompt);
    const int n_predict = pkg->model_ptr->m_max_inference_ctx;
    tokens.reserve(n_predict);

    std::vector<std::pair<double, int>> logits_probs;

    const float temp = 0.9f;
    const int top_k = 50;
    const int eot_token = pkg->tokenizer_ptr->m_eos_token;
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
        const int logits_size = logits.numel(); 

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
        // std::cout << pred_token << "\n";


        const char* piece = pkg->tokenizer_ptr->decode(prev_token, pred_token);
        {
            callback_function(piece);
        }

        tokens.push_back(pred_token);
    }

    if (!reached_eot) {
        callback_function("<endoftext>");
    }
}
