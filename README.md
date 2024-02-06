# NanoChat.ai
![alt text](./assets/tinyllama.jpeg)

**NanoChat.ai** is a cross-platform GUI application built on Electron that allows you to chat with
small but powerful chat language models. The inference is implemented in pure C++
and is fast even on lower-end devices, e.g 4GB RAM and dual-core CPU. Currently existing
models are [Zephyr1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b) and
[TinyLlama-1.1B-Chat-v0.4](https://github.com/jzhang38/TinyLlama). All models have
Float-16 and 8-bit and 4-bit quantized versions.

## Build locally
To build NanoChat, the following dependencies are required:
- [Node.js and npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) version v20.11.0 and v10.2.4 (or higher) respectively.
- [Electron](https://www.electronjs.org/) which can be installed via npm by the following command:
   `npm install --save-dev electron@latest`.
- [Cmake](https://cmake.org/download/#latest) version 3.25 or higher.

## Install and Run NanoChat.
Once you have the required dependecies run the following commands to install and launch the app and enjoy!

```
git clone https://github.com/iangitonga/nanochat.ai.git
cd nanochat/
npm install
npm start
```
