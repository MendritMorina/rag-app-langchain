import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import path from "node:path";

class PdfQA {
  constructor({
    chatModel,
    embeddingModel,
    pdfPath,
    chunkSize,
    chunkOverlap,
    searchType = "similarity",
    kDocuments = 2,
  }) {
    this.chatModel = chatModel;
    this.embeddingModel = embeddingModel;
    this.pdfPath = pdfPath;
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.searchType = searchType;
    this.kDocuments = kDocuments;
  }

  async init() {
    this.initChatModel();
    await this.loadDocuments();
    await this.splitDocuments();
    this.initEmbeddings();
    await this.createVectorStore();
    this.createRetriever();
    await this.createChain();

    return this;
  }

  initChatModel() {
    console.log("Loading model...");
    this.llm = new Ollama({ model: this.chatModel });
  }

  async loadDocuments() {
    console.log("Loading PDF...");
    const loader = new PDFLoader(this.pdfPath);
    this.documents = await loader.load();
    console.log(`Loaded ${this.documents.length} PDF page(s).`);
  }

  async splitDocuments() {
    console.log("Splitting documents...");
    const splitter = new CharacterTextSplitter({
      separator: " ",
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
    });

    this.chunks = await splitter.splitDocuments(this.documents);
    console.log(`Created ${this.chunks.length} text chunk(s).`);
  }

  initEmbeddings() {
    this.embeddings = new OllamaEmbeddings({ model: this.embeddingModel });
  }

  async createVectorStore() {
    console.log("Creating document embeddings...");
    this.vectorStore = await MemoryVectorStore.fromDocuments(
      this.chunks,
      this.embeddings,
    );
  }

  createRetriever() {
    console.log("Creating retriever...");
    this.retriever = this.vectorStore.asRetriever({
      k: this.kDocuments,
      searchType: this.searchType,
    });
  }

  async createChain() {
    console.log("Creating RAG chain...");

    const prompt = ChatPromptTemplate.fromTemplate(`
Answer the question using only the supplied context.
If the answer is not in the context, say that you do not know.

Context:
{context}

Question:
{input}
`);

    const documentChain = await createStuffDocumentsChain({
      llm: this.llm,
      prompt,
    });

    this.chain = await createRetrievalChain({
      retriever: this.retriever,
      combineDocsChain: documentChain,
    });
  }

  async ask(question) {
    return this.chain.invoke({ input: question });
  }
}

async function main() {
  const pdfQa = new PdfQA({
    chatModel: "gemma3:1b",
    embeddingModel: "nomic-embed-text",
    pdfPath: path.join(import.meta.dirname, "sample.pdf"),
    chunkSize: 1000,
    chunkOverlap: 0,
    searchType: "similarity",
    kDocuments: 2,
  });

  await pdfQa.init();

  const result = await pdfQa.ask(
    "How do we add a custom file type in PyCharm?",
  );

  console.log("\nAnswer:");
  console.log(result.answer);
}

main().catch((error) => {
  console.error("Failed to run PDF Q&A:", error.message);
  process.exitCode = 1;
});
