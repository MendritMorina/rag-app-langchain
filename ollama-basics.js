import { Ollama } from "@langchain/ollama";

class PdfQA {
  constructor({ model }) {
    this.model = model;
  }

  async init() {
    await this.initChatModel();

    return this;
  }

  async initChatModel() {
    console.log("Loading model...");

    this.llm = new Ollama({ model: this.model });

    const response = await this.llm.invoke("What's the capital of Kosovo?");
    console.log(response);
  }
}

await new PdfQA({ model: "gemma3:1b" }).init();
