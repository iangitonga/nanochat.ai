const os = require("node:os");
const path = require("node:path");
const fs = require("node:fs");
const https = require('node:https');
const bootstrap = require("bootstrap");


const inference_worker = new Worker("./inference_worker.js");
const load_worker = new Worker("./load_worker.js");

const global_state = {
    inference_pkg_id: null,
    processed_prompts: 0,
    model_download_cancelled: false,
    loaded_models: []
};

const send_button = document.getElementById("chat-submit-btn");
const chat_input_form = document.getElementById("chat-input-form");
const chat_input = document.getElementById("chat-input");
const chat_display = document.getElementById("chat-display");
const model_selector = document.getElementById("model-selector");
const model_format_selector = document.getElementById("model-format-selector");


const model_urls = {
    minicpm: {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/minicpm.fp16.gten",
        "q8"  : "https://huggingface.co/iangitonga/gten/resolve/main/minicpm.q8.gten",
        "q4"  : "https://huggingface.co/iangitonga/gten/resolve/main/minicpm.q4.gten"
    },
    zephyr: {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/zephyr.fp16.gten",
        "q8"  : "https://huggingface.co/iangitonga/gten/resolve/main/zephyr.q8.gten",
        "q4"  : "https://huggingface.co/iangitonga/gten/resolve/main/zephyr.q4.gten"
    },
    tinyllama: {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/tinyllama.fp16.gten",
        "q8"  : "https://huggingface.co/iangitonga/gten/resolve/main/tinyllama.q8.gten",
        "q4"  : "https://huggingface.co/iangitonga/gten/resolve/main/tinyllama.q4.gten"
    }
};


const tokenizer_paths = {
    minicpm: path.join(__dirname, "..", "assets", "tokenizers", 'minicpm_tokenizer.bin'),
    zephyr: path.join(__dirname, "..", "assets", "tokenizers", 'zephyr_tokenizer.bin'),
    tinyllama: path.join(__dirname, "..", "assets", "tokenizers", 'tinyllama_tokenizer.bin')
};

const show_error_alert = (message) => {
    const alert_html = `
        <div class="alert alert-dismissible alert-danger m-3 fade show" id="error-alert" role="alert">
          ${message}
          <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `
    document.getElementById("main").insertAdjacentHTML("afterbegin", alert_html)
};

const show_download_progress = (cursize_mb, totsize_mb) => {
    document.getElementById("download-modal-accumsize").innerText = cursize_mb;
    document.getElementById("download-progbar").style.width = `${(cursize_mb / totsize_mb * 100).toFixed(0)}%`;
};


const get_actual_download_url = (model_download_url, callback) => {
    const request = https.get(model_download_url, (response) => {
        let raw_data = "";

        response.setEncoding('utf8');
        response.on('data', (chunk) => {
            raw_data += chunk;
        });

        response.on('end', () => {
            const actual_model_download_url = raw_data.substring(22);
            callback(actual_model_download_url);
        });

    }).on("error", (err) => {
        console.log(`failed to fetch actual dl url for: ${model_download_url}`);
        callback("", err);
    });
};


document.getElementById("model-download-cancel").addEventListener("click", (event) => {
    event.preventDefault();

    global_state.model_download_cancelled = true;
});


const show_download_modal = (model_name, model_format) => {
    // show download modal.
    document.getElementById("model-download-modal-title").innerText = `${model_name} (${model_format})`
    const model_download_modal = document.getElementById("model-download-modal");
    const modal = bootstrap.Modal.getOrCreateInstance(model_download_modal);
    modal.show();
}

const hide_download_modal = () => {
    const model_download_modal = document.getElementById("model-download-modal");
    const modal = bootstrap.Modal.getInstance(model_download_modal);
    modal.hide();
}

const download_model = (model_name, model_format, callback) => {
    const model_download_url = model_urls[model_name][model_format];
    console.log(model_download_url);
    show_download_modal(model_name, model_format);

    // CREATE DOWNLOAD AND TEMPORARY DIRECTORIES.
    const home_path = os.homedir();
    const model_name_id = `${model_name}.${model_format}.gten`;
    // Create model path.
    const model_dir_path = path.join(home_path, ".cache", "nanochatllms", "models");
    if (!fs.existsSync(model_dir_path)) {
        const ret = fs.mkdirSync(model_dir_path, { recursive: true });
        if (!ret) {
            const err = new Error(`failed to create: ${model_dir_path}`);
            hide_download_modal();
            callback(err);
        }
    }

    // create temporary model download path.
    const temp_dir_path = path.join(home_path, ".cache", "nanochatllms", "temp");
    if (!fs.existsSync(temp_dir_path)) {
        const ret = fs.mkdirSync(temp_dir_path, { recursive: true });
        if (!ret) {
            const err = new Error(`failed to create: ${temp_dir_path}`);
            hide_download_modal();
            callback(err);
        }
    }

    // DOWNLOAD
    {
        // Check for status code?
        // HuggingFace creates a temporary download link which it redirects to, we need
        // to first make a request to retrieve the actual download link.
        get_actual_download_url(model_download_url, (actual_model_dl_url, err) => {
            if (err) {
                hide_download_modal();
                callback(err);
                return;
            }

            const temp_file_path = path.join(temp_dir_path, model_name_id);
            const temp_stream = fs.createWriteStream(temp_file_path, {flags: "w"});

            // reset
            global_state.model_download_cancelled = false;

            const request = https.get(actual_model_dl_url, (response) => {
                const totsize_bytes = parseInt(response.headers['content-length'], 10);
                const totsize_mb = (totsize_bytes / 1000000).toFixed(0);
                let cursize_bytes = 0;

                document.getElementById("download-modal-totsize").innerText = totsize_mb;

                const reset_connection = () => {
                    fs.rmSync(temp_file_path);
                    // Ensures that no more I/O activity happens on this socket. Destroys the stream and closes the connection.
                    response.destroy();
                    const err = new Error(`failed to create: ${temp_dir_path}`);
                    console.log("error: ", err);
                    hide_download_modal();
                };

                response.on('data', (chunk) => {
                    if (global_state.model_download_cancelled) {
                        reset_connection();
                        const err = new Error(`failed to create: ${temp_dir_path}`);
                        callback(err);
                    } else {
                        cursize_bytes += chunk.length;
                        show_download_progress((cursize_bytes/1000000).toFixed(0), totsize_mb);
                    }
                });

                response.on("error", (err) => {
                    reset_connection();
                    callback(err);
                });

                response.on('end', () => {
                    console.log("Download complete");
                    // Update UI
                    hide_download_modal();

                    // Move from temp to dest.
                    const model_file_path = path.join(model_dir_path, model_name_id);
                    fs.renameSync(temp_file_path, model_file_path);

                    callback(null);
                });

                response.pipe(temp_stream);
            }).on("error", (err) => {
                hide_download_modal();

                callback(err);
            });
            
        });
    }
};


const _load_model = (model_name, model_format, model_path, callback) => {
    const tok_path = tokenizer_paths[model_name];
    const n_ctx = 800;

    const data = {
        "model_name": model_name,
        "model_dtype": model_format,
        "model_path": model_path,
        "tokenizer_path": tok_path,
        "n_ctx": n_ctx
    };
    load_worker.postMessage(data);

    load_worker.onmessage = (event) => {
        console.log("Load worker received data: ", event.data);
        global_state.inference_pkg_id = event.data;
        global_state.loaded_models.push(`${model_name}.${model_format}`);
        callback(null);
    };

    load_worker.onerror = (event) => {
        console.log("load worker error.")
        console.log(event.message, event);
        const err = new Error("Load worker error")
        callback(err);
    }
}


const load_model = (model_name, model_format, callback) => {
    if (global_state.loaded_models.includes(`${model_name}.${model_format}`)) {
        console.log("model is loaded");
        callback(null);
        return;
    }

    gten_assert(model_urls.hasOwnProperty(model_name), "Unexpected model.");
    gten_assert(model_urls[model_name].hasOwnProperty(model_format), "Unexpected model format.");

    const home_path = os.homedir();
    const model_name_id = `${model_name}.${model_format}.gten`;
    const model_path = path.join(home_path, ".cache", "nanochatllms", "models", model_name_id);

    console.log(model_path);

    if (!fs.existsSync(model_path))
    {
        download_model(model_name, model_format, (err) => {
            if (err) {
                console.log(err);
                callback(err);
                return;
            }
            else {
                //load modal.
                _load_model(model_name, model_format, model_path, callback);
            }
        });
    } else {
        console.log("model path exists");
        _load_model(model_name, model_format, model_path, callback);
    }
};


chat_input_form.addEventListener("submit", (event) => {
    event.preventDefault();

    // Important: trim trailing ans leading whitespace if any.
    const prompt_text = chat_input.value.trim();

    if (prompt_text == "") {
        return;
    }
    const selected_model = model_selector.value;
    const selected_model_format = model_format_selector.value;

    send_button.setAttribute("disabled", true);
    chat_input.value = "";


    appendMessage(PERSON_NAME, "left", prompt_text);
    appendMessage(BOT_NAME, "left", "");

    load_model(selected_model, selected_model_format, (err) => {
        if (err) {
            send_button.removeAttribute("disabled");
            // stop_bot_msg_loader(global_state.processed_prompts);
            document.getElementById(`msg1-${global_state.processed_prompts}`).remove();
            show_error_alert("Model loading failed. Check your internet connection and try again.");
            console.log(err);
        } else {
            // Inference ready.
            perform_inference(prompt_text);
        }
    });

});


const BOT_NAME = "BOT";
const PERSON_NAME = "YOU";

const perform_inference = (prompt_text) => {
    if (global_state.inference_pkg_id) {
        const post_data = {
            inference_pkg_id: global_state.inference_pkg_id,
            prompt: prompt_text,
        };

        inference_worker.postMessage(post_data);
    } else {
        console.log("Error: inference_pkg_id missing");
        send_button.removeAttribute("disabled");
    }
}

let last_was_linebreak = false;
inference_worker.onmessage = (event) => {
    const current_bot_msg_id = `bot-msg-${global_state.processed_prompts}`;
    const current_bot_msg = document.getElementById(current_bot_msg_id);
    current_bot_msg.parentElement.querySelector(".msg-loader").classList.add("d-none");

    let pred_word = event.data;
    if (pred_word != "<endoftext>") {
        if (pred_word.includes("\n") && !last_was_linebreak) {
            pred_word = "<br/><br/>";
            last_was_linebreak = true;
        } else {
            last_was_linebreak = false;
        }
        current_bot_msg.insertAdjacentHTML("beforeend", pred_word);
        // current_bot_msg.scrollIntoView(/*alignToTop=*/false);
        window.scrollTo(0, document.body.scrollHeight);

    } else {
        last_was_linebreak = false;
        global_state.processed_prompts = global_state.processed_prompts + 1;
        send_button.removeAttribute("disabled");
    }
}


function appendMessage(name, side, text) {
  const id = global_state.processed_prompts;
  const element_id = (name === "BOT") ? `bot-msg-${id}` : `prompt-${id}`;

  let msgHTML;
  if (name == "BOT") {
      msgHTML = `
        <div class="msg ${side}-msg" id="msg1-${id}">
          <div class="msg-bubble msg-${name.toLowerCase()}">
            <div class="msg-loader" style="width: 100%;height: 5px;border-radius: 10px;"></div>
            <div class="msg-info">
              <div class="msg-info-name">${name}</div>
            </div>

            <div class="msg-text" id="${element_id}">${text}</div>
          </div>
        </div>
      `;
  } else {
    msgHTML = `
        <div class="msg ${side}-msg">
          <div class="msg-bubble msg-${name.toLowerCase()}">
            <div class="msg-info">
              <div class="msg-info-name">${name}</div>
            </div>

            <div class="msg-text" id="${element_id}">${text}</div>
          </div>
        </div>
      `;
  }

  chat_display.insertAdjacentHTML("beforeend", msgHTML);
    document.getElementById(element_id).scrollIntoView(/*alignToTop=*/true);
    
}

document.getElementById("dark-mode-switch").addEventListener("change", (event) => {
    event.preventDefault();

    const body = document.getElementById("body");
    if (body.getAttribute("data-bs-theme") === "light") {
        body.setAttribute("data-bs-theme", "dark");
    } else {
        body.setAttribute("data-bs-theme", "light");
    }
});


function gten_assert(condition, message) {
    if (!condition) {
        throw new Error(message || "Assertion failed");
    }
}

// const model_selector = document.getElementById("model-selector");
// const model_format_selector = document.getElementById("model-format-selector");

const model_size = {
    minicpm: {
        "fp16": "FP-16 (5.5GB)",
        "q8"  : "8-bit (2.9GB)",
        "q4"  : "4-bit (1.5GB)"
    },
    zephyr: {
        "fp16": "FP-16 (3.3GB)",
        "q8"  : "8-bit (1.8GB)",
        "q4"  : "4-bit (0.9GB)"
    },
    tinyllama: {
        "fp16": "FP-16 (2.2GB)",
        "q8"  : "8-bit (1.2GB)",
        "q4"  : "4-bit (0.6GB)"
    }
}

model_selector.addEventListener("change", (event) => {
    const selected_model = event.target.value;

    document.getElementById(`fp16-option`).innerText = model_size[selected_model]["fp16"];
    document.getElementById(`q8-option`).innerText = model_size[selected_model]["q8"];
    document.getElementById(`q4-option`).innerText = model_size[selected_model]["q4"];
});
