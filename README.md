# PDF RAG with LangChain and Ollama

A small JavaScript project that answers questions about a local PDF using
LangChain and locally hosted Ollama models.

The application uses Retrieval-Augmented Generation (RAG): it reads a PDF,
splits the text into chunks, creates embeddings, retrieves the most relevant
chunks, and gives them to the language model as context for its answer.

## How it works

```text
PDF
 ↓
PDFLoader
 ↓
CharacterTextSplitter
 ↓
OllamaEmbeddings
 ↓
MemoryVectorStore
 ↓
Retriever
 ↓
Ollama language model
 ↓
Answer based on the PDF
```

## Requirements

- Node.js 20 or newer
- Ollama running locally
- The following Ollama models:
  - `gemma3:1b` for generating answers
  - `nomic-embed-text` for creating embeddings

Install the models with:

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

## Installation

Clone the repository and install its dependencies:

```bash
git clone git@github.com:MendritMorina/rag-app-langchain.git
cd rag-app-langchain
npm install
```

If npm reports a peer-dependency conflict, use:

```bash
npm install --legacy-peer-deps
```

## Usage

Make sure Ollama is running, then execute:

```bash
npm start
```

You can also run the entry file directly:

```bash
node load-pdf.js
```

To check the JavaScript syntax without running the models:

```bash
npm test
```

## Using another PDF

Place the PDF in the project directory and change this line in
`load-pdf.js`:

```js
const pdfDocument = "./sample.pdf";
```

Change the question near the bottom of the same file:

```js
const result = await pdfQaChain.invoke({
  input: "Ask a question about your PDF here",
});
```

The model is instructed to answer only from the retrieved PDF context. If the
document does not contain the requested information, the expected answer is
that it does not know.

## Configuration

The main options are passed to `PdfQA` in `load-pdf.js`:

```js
const pdfQa = await new PdfQA({
  model: "gemma3:1b",
  pdfDocument: "./sample.pdf",
  chunkSize: 1000,
  chunkOverlap: 0,
  searchType: "similarity",
  kDocuments: 2,
}).init();
```

- `model`: Ollama model used to generate the answer.
- `pdfDocument`: PDF path relative to `load-pdf.js`.
- `chunkSize`: maximum size of each text chunk.
- `chunkOverlap`: amount of repeated text between adjacent chunks.
- `searchType`: vector-store retrieval strategy.
- `kDocuments`: number of relevant chunks supplied to the language model.

The embedding model is configured separately:

```js
this.embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
});
```

## Main dependencies

- LangChain for the RAG pipeline
- `@langchain/ollama` for local generation and embeddings
- `PDFLoader` and `pdf-parse` for extracting PDF text
- `MemoryVectorStore` for in-memory similarity search

Because the vector store is held in memory, its contents are recreated every
time the application starts.

## License

ISC
