const addon = require("../build/Release/backend.node");


onmessage = (event) => {
    console.log('Worker: Message received from main script');
    console.log("message: ", event.data);

    var callback_function = (pred_word) => {
        // console.log(pred_word);
        postMessage(pred_word);
    };

    addon.perform_inference(event.data.inference_pkg_id, event.data.prompt, callback_function);
}
