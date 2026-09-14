import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import path from "node:path";

class PdfQA {
  constructor({
    model,
    pdfDocument,
    chunkSize,
    chunkOverlap,
    searchType = "similarity",
    kDocuments = 2,
  }) {
    this.model = model;
    this.pdfDocument = pdfDocument;
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.searchType = searchType;
    this.kDocuments = kDocuments;
  }

  async init() {
    this.initChatModel();
    await this.loadDocuments();
    await this.splitDocuments();

    this.embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });

    await this.createVectoreStore();
    this.createRetriever();
    await this.createChain();
    return this;
  }

  async initChatModel() {
    console.log("Loading model...");

    this.llm = new Ollama({ model: this.model });
  }

  async loadDocuments() {
    console.log("Loading PDFs...");

    const pdfLoader = new PDFLoader(
      path.join(import.meta.dirname, this.pdfDocument),
    );

    this.documents = await pdfLoader.load();
  }

  async splitDocuments() {
    console.log("Splitting documents...");

    const textSplitter = new CharacterTextSplitter({
      separator: " ",
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
    });

    this.texts = await textSplitter.splitDocuments(this.documents);
  }

  async createVectoreStore() {
    console.log("Creating document embeddings...");

    this.db = await MemoryVectorStore.fromDocuments(
      this.texts,
      this.embeddings,
    );
  }

  createRetriever() {
    console.log("Initialize vector store retriever...");
    this.retriever = this.db.asRetriever({
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

  queryChain() {
    return this.chain;
  }
}

const pdfDocument = "./sample.pdf";

const pdfQa = await new PdfQA({
  model: "gemma3:1b",
  pdfDocument,
  chunkSize: 1000,
  chunkOverlap: 0,
  searchType: "similarity",
}).init();

// Load documents

// console.log(pdfQa);
// console.log(pdfQa.documents.length);
// console.log(pdfQa.documents[0].pageContent);

// Split documents
// console.log(pdfQa.texts);
// console.log(pdfQa.texts.length);

// Embeddings
// console.log(pdfQa.db.embeddings);
// console.log(pdfQa.db.embeddings.model);
// console.log(pdfQa.db.memoryVectors.length);

// const similaritySearchResults = await pdfQa.db.similaritySearch(
//   "File type associations",
//   2,
// );

// console.log("Document pages related to our query:");

// for (const doc of similaritySearchResults) {
//   console.log(JSON.stringify(doc.metadata.loc, null, 2));
// }

// Retriever
// const relevantDocuments = await pdfQa.retriever.invoke(
//   "What can you do with AI Assistant",
// );
// console.log(relevantDocuments);
// console.log(relevantDocuments[0].pageContent);
// console.log(relevantDocuments[0].metadata);

// Chain
const pdfQaChain = pdfQa.queryChain();

const result = await pdfQaChain.invoke({
  input: "How do we add a custom file type in PyCharm?",
});

console.log("\nAnswer:");
console.log(result.answer);
