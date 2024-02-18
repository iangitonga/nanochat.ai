#include <node_api.h>

#include "api.h"


namespace js_api {

#define ASSERT_NAPI_STATUS(env, status, error_msg)   \
        if (status != napi_ok) {                     \
          std::cout << "error: " << error_msg << "\n"; \
          napi_throw_error(env, "", error_msg);      \
          return nullptr;                            \
        }


// Load model and tokenizer.
// inp: model_name, model_type, model_path, tokenizer_path, n_ctx
napi_value api_init_inference_package(napi_env env, napi_callback_info info) {
    const size_t expected_inp_argc = 5;
    size_t inp_argc = expected_inp_argc;
    napi_value inp_args[expected_inp_argc];

    napi_status status = napi_get_cb_info(env, info, &inp_argc, inp_args, NULL, NULL);
    ASSERT_NAPI_STATUS(env, status, "fn `napi_get_cb_info` failed.");

    { // INPUT ARGS ERROR CHECKING
        // TODO: Improve error reporting.
        if (inp_argc < expected_inp_argc) {
            napi_throw_type_error(env, nullptr, "api_init_inference_package: Incorrect number of arguments");
            return nullptr;
        }

        napi_valuetype arg0_type;
        status = napi_typeof(env, inp_args[0], &arg0_type);
        ASSERT_NAPI_STATUS(env, status, "fn napi_typeof failed.");

        if (arg0_type != napi_string) {
            napi_throw_type_error(env, nullptr, "api_init_inference_package: arg 0 has incorrect type.");
            return nullptr;
        }

        napi_valuetype arg1_type;
        status = napi_typeof(env, inp_args[1], &arg1_type);
        ASSERT_NAPI_STATUS(env, status, "fn napi_typeof failed.");

        if (arg1_type != napi_string) {
            napi_throw_type_error(env, nullptr, "api_init_inference_package: arg 1 has incorrect type.");
            return nullptr;
        }

        napi_valuetype arg2_type;
        status = napi_typeof(env, inp_args[2], &arg2_type);
        ASSERT_NAPI_STATUS(env, status, "fn napi_typeof failed.");

        if (arg2_type != napi_string) {
            napi_throw_type_error(env, nullptr, "api_init_inference_package: arg 2 has incorrect type.");
            return nullptr;
        }

        napi_valuetype arg3_type;
        status = napi_typeof(env, inp_args[3], &arg3_type);
        ASSERT_NAPI_STATUS(env, status, "fn napi_typeof failed.");

        if (arg3_type != napi_string) {
            napi_throw_type_error(env, nullptr, "api_init_inference_package: arg 3 has incorrect type.");
            return nullptr;
        }

        napi_valuetype arg4_type;
        status = napi_typeof(env, inp_args[4], &arg4_type);
        ASSERT_NAPI_STATUS(env, status, "fn napi_typeof failed.");

        if (arg4_type != napi_number) {
            napi_throw_type_error(env, nullptr, "api_init_inference_package: arg 4 has incorrect type.");
            return nullptr;
        }
    }

    const int string_bufsize = 1024;
    char string_buf[string_bufsize];
    size_t string_size;
    status = napi_get_value_string_utf8(env, inp_args[0], string_buf, string_bufsize, &string_size);
    ASSERT_NAPI_STATUS(env, status, "fn napi_get_value_string_utf8 failed.");

    const std::string model_name{string_buf};

    status = napi_get_value_string_utf8(env, inp_args[1], string_buf, string_bufsize, &string_size);
    ASSERT_NAPI_STATUS(env, status, "fn napi_get_value_string_utf8 failed.");

    const std::string model_type_id{string_buf};
    
    Dtype model_dtype;
    if (model_type_id == "fp16") {model_dtype = kFloat16; }
    else if (model_type_id == "q8") {model_dtype = kQint8; } 
    else {model_dtype = kQint4; } 

    status = napi_get_value_string_utf8(env, inp_args[2], string_buf, string_bufsize, &string_size);
    ASSERT_NAPI_STATUS(env, status, "fn napi_get_value_string_utf8 failed.");
    const std::string model_path{string_buf};

    status = napi_get_value_string_utf8(env, inp_args[3], string_buf, string_bufsize, &string_size);
    ASSERT_NAPI_STATUS(env, status, "fn napi_get_value_string_utf8 failed.");
    const std::string tokenizer_path{string_buf};

    int n_ctx;
    status = napi_get_value_int32(env, inp_args[4], &n_ctx);
    ASSERT_NAPI_STATUS(env, status, "fn napi_get_value_int32 failed.");

    std::cout<< "mname: " << model_name << "\n";
    std::cout<< "mdtyp: " << model_type_id << "\n"; 
    std::cout<< "mdirn: " << model_path << "\n"; 
    std::cout<< "mtokp: " << tokenizer_path << "\n"; 
    std::cout<< "mnctx: " << n_ctx << "\n"; 

    void* pkg_ptr = init_inference_package(model_name, model_dtype, model_path, tokenizer_path, n_ctx);
    // TODO: Could 'napi_create_external' be used to carry the pointer?
    const uint64_t ptr_int = (uint64_t)pkg_ptr;

    napi_value ret_value;
    status = napi_create_bigint_uint64(env, ptr_int, &ret_value);
    ASSERT_NAPI_STATUS(env, status, "fn napi_create_bigint_uint64 failed.");

    return ret_value;
}


// input: inference_pkg_ptr, prompt, callback_function = (pred_text) => {...} 
napi_value api_perform_inference(napi_env env, napi_callback_info info) {
    const size_t expected_inp_argc = 3;
    size_t inp_argc = expected_inp_argc;
    napi_value inp_args[expected_inp_argc];

    napi_status status = napi_get_cb_info(env, info, &inp_argc, inp_args, NULL, NULL);
    ASSERT_NAPI_STATUS(env, status, "fn `napi_get_cb_info` failed.");

    { // INPUT ARGS ERROR CHECKING
        // TODO: Improve error reporting.
        if (inp_argc < expected_inp_argc) {
            napi_throw_type_error(env, nullptr, "api_perform_inference: Incorrect number of arguments");
            return nullptr;
        }

        napi_valuetype arg0_type;
        status = napi_typeof(env, inp_args[0], &arg0_type);
        ASSERT_NAPI_STATUS(env, status, "fn napi_typeof failed.");

        if (arg0_type != napi_bigint) {
            napi_throw_type_error(env, nullptr, "api_perform_inference: arg 0 has incorrect type.");
            return nullptr;
        }

        napi_valuetype arg1_type;
        status = napi_typeof(env, inp_args[1], &arg1_type);
        ASSERT_NAPI_STATUS(env, status, "fn napi_typeof failed.");

        if (arg1_type != napi_string) {
            napi_throw_type_error(env, nullptr, "api_perform_inference: arg 1 has incorrect type.");
            return nullptr;
        }

        napi_valuetype arg2_type;
        status = napi_typeof(env, inp_args[2], &arg2_type);
        ASSERT_NAPI_STATUS(env, status, "fn napi_typeof failed.");

        if (arg2_type != napi_function) {
            napi_throw_type_error(env, nullptr, "api_perform_inference: arg 2 has incorrect type.");
            return nullptr;
        }
    }

    // PKG PTR
    uint64_t inference_pkg_ptr_int;
    bool conversion_is_lossless;
    status = napi_get_value_bigint_uint64(env, inp_args[0], &inference_pkg_ptr_int, &conversion_is_lossless);
    ASSERT_NAPI_STATUS(env, status, "fn napi_get_value_bigint_uint64 failed.");
    assert(conversion_is_lossless);
    void* inference_pkg_ptr = reinterpret_cast<void*>(inference_pkg_ptr_int);

    // PROMPT
    const int prompt_bufsize = 6000; // MAX_PROMPT_SIZE.
    char prompt_buf[prompt_bufsize];
    size_t prompt_size;
    status = napi_get_value_string_utf8(env, inp_args[1], prompt_buf, prompt_bufsize, &prompt_size);
    ASSERT_NAPI_STATUS(env, status, "fn napi_get_value_string_utf8 failed.");
    if (prompt_size > prompt_bufsize) { 
        napi_throw_error(env, nullptr, "api_perform_inference: Prompt size too large.");
        return nullptr;
    }
    std::string prompt{prompt_buf};

    // CB_FUNCTION
    napi_value callback_function = inp_args[2];

    const auto inference_cb = [env, callback_function](const char* pred_word) {
        const size_t cb_argc = 1;
        napi_value cb_argv[cb_argc];

        napi_status status = napi_create_string_utf8(env, pred_word, NAPI_AUTO_LENGTH, cb_argv); // event
        if (status != napi_ok) { napi_throw_error(env, "", "fn napi_create_string_utf8 failed."); }

        napi_value global;
        status = napi_get_global(env, &global);
        if (status != napi_ok) { napi_throw_error(env, "", "fn napi_get_global failed."); }

        napi_value result;
        status = napi_call_function(env, global, callback_function, cb_argc, cb_argv, &result);
        if (status != napi_ok) { napi_throw_error(env, "", "fn napi_call_function failed."); }
    };

    // INFERENCE
    const std::string model_name = reinterpret_cast<InferencePackage*>(inference_pkg_ptr)->model_name;

    perform_inference(inference_pkg_ptr, prompt, inference_cb);

    return nullptr;
}


napi_value init(napi_env env, napi_value exports) {
    const napi_property_descriptor desc[] = {
        {"init_inference_engine", 0, api_init_inference_package, 0, 0, 0, napi_default, 0},
        {"perform_inference", 0, api_perform_inference, 0, 0, 0, napi_default, 0},
    };

    const size_t desc_size = sizeof(desc) / sizeof(*desc);
    napi_status status = napi_define_properties(env, exports, desc_size, desc);
    ASSERT_NAPI_STATUS(env, status, "fn napi_define_properties failed.");

    return exports;
}


NAPI_MODULE(NODE_GYP_MODULE_NAME, init)

}  // namespace js_api