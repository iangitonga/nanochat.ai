# NanoChat.ai
![alt text](./nanochat.png)

**NanoChat.ai** is a cross-platform GUI application built on Electron that allows you to chat with
small but powerful chat language models. The inference is implemented in pure C++
and is fast even on lower-end devices, e.g 4GB RAM and dual-core CPU thanks to 8-bit and 4-bit inference.
Currently existing models:
   1. [MiniCPM](https://huggingface.co/openbmp/MiniCPM-2B-dpo-fp16) [license](ttps://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md)
   2. [Zephyr1.6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)
   3. [TinyLlama-1.1B-Chat-v0.4](https://github.com/jzhang38/TinyLlama).

## Build locally
To build NanoChat, the following dependencies are required:
- [Node.js and npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) version v20.11.0 and v10.2.4 (or higher) respectively.
- [Electron](https://www.electronjs.org/) which can be installed via npm by the following command:
   `npm install --save-dev electron@latest`.
- [Cmake](https://cmake.org/download/#latest) version 3.0 or higher.

## Install and Run NanoChat.
Once you have the required dependecies run the following commands to install and launch the app and enjoy!

```
git clone https://github.com/iangitonga/nanochat.ai.git
cd nanochat.ai/
npm install
npm start
```
