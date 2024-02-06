const addon = require("../build/Release/backend.node");


onmessage = (event) => {
    console.log('Worker: Message received from main script');
    console.log("message: ", event.data);

    const data = event.data;

    const model_name = data.model_name;
    const model_dtype = data.model_dtype;
    const model_path = data.model_path;
    const tokenizer_path = data.tokenizer_path;
    const n_ctx = data.n_ctx;

    const result = addon.init_inference_engine(model_name, model_dtype, model_path, tokenizer_path, n_ctx);
	console.log(result);

	postMessage(result);
}





