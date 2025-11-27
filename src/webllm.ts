import appConfig from "./app-config";
import * as webllm from "@mlc-ai/web-llm";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { Document } from "@langchain/core/documents";
import { formatDocumentsAsString } from "langchain/util/document";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import List from "list.js";
import { JSDOM } from 'jsdom';
import DOMPurify from 'dompurify';
import { marked } from 'marked';
import products from './products.json';

function getElementAndCheck(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (element == null) {
    throw Error("Cannot find element " + id);
  }
  return element;
}

class ChatUI {
  private uiChat!: HTMLElement;
  private uiChatInput!: HTMLInputElement;
  private uiChatInfoLabel!: HTMLLabelElement;
  private engine!: webllm.MLCEngineInterface;
  private config: webllm.AppConfig = appConfig;
  private embeddingModel: string = "snowflake-arctic-embed-m-q0f32-MLC-b32"; //snowflake-arctic-embed-m-q0f32-MLC-b4
  private llmModel: string = "gemma-2-2b-it-q4f32_1-MLC-1k";
  private chatLoaded = false;
  private requestInProgress = false;
  private chatHistory: webllm.ChatCompletionMessageParam[] = [];
  //
  private uiChatSendButton!: HTMLElement;
  private uiChatResetButton!: HTMLElement;
  private uiChatDownloadButton!: HTMLElement;
  private uiChatProgress!: HTMLElement;
  private uiChatProgressbar!: HTMLDivElement
  // We use a request chain to ensure that
  // all requests send to chat are sequentialized
  private chatRequestChain: Promise<void> = Promise.resolve();

  private showProgress(show = false) {
    if (show) {
      this.uiChatProgress.classList.remove("hidden");
      this.uiChatDownloadButton.classList.add("hidden");
    } else {
      this.uiChatDownloadButton.classList.remove("hidden");
      this.uiChatProgress.classList.add("hidden");
    }
  }

  private enableChatInput(enable = true) {
    if (enable) {
      this.uiChatInput.removeAttribute("disabled");
      this.uiChatSendButton.removeAttribute("disabled");
      this.uiChatResetButton.removeAttribute("disabled");
    } else {
      this.uiChatInput.setAttribute("disabled", "true");
      this.uiChatSendButton.setAttribute("disabled", "true");
      this.uiChatResetButton.setAttribute("disabled", "true");
    }
  }

  initProgressCallback = (report: {text: string}) => {
    // Update the progressbar
    let progress_percentage = report.text.match(/(\d+)%/)?.[0];
    if (progress_percentage) {
      this.uiChatProgressbar!.style.width = progress_percentage;
    }
    // Update the init message in the chat box
    this.updateLastMessage("warning", report.text);
  }

  /**
   * An asynchronous factory constructor since we need to await getMaxStorageBufferBindingSize();
   * this is not allowed in a constructor (which cannot be asynchronous).
   */
  public static CreateAsync = async (/*engine: webllm.MLCEngineInterface*/) => {
    const chatUI = new ChatUI();
    chatUI.engine = new webllm.MLCEngine({ appConfig }); //engine;
    // get the elements
    chatUI.uiChat = getElementAndCheck("chat-box");
    chatUI.uiChatInput = getElementAndCheck("chat-user-input") as HTMLInputElement;
    chatUI.uiChatInfoLabel = getElementAndCheck(
      "chat-info-label",
    ) as HTMLLabelElement;
    chatUI.uiChatSendButton = getElementAndCheck("chat-send-btn");
    chatUI.uiChatResetButton = getElementAndCheck("chat-reset-btn");
    chatUI.uiChatDownloadButton = getElementAndCheck("chat-download-models");
    chatUI.uiChatProgress = getElementAndCheck("chat-download-progressbar-container");
    chatUI.uiChatProgressbar = getElementAndCheck(
      "chat-download-progressbar",
    ) as HTMLDivElement;
    // register event handlers
    chatUI.uiChatResetButton.onclick = () => {
      chatUI.onReset();
    };
    chatUI.uiChatSendButton.onclick = () => {
      chatUI.onGenerate();
    };
    chatUI.uiChatDownloadButton.onclick = async () => {
      chatUI.uiChatDownloadButton.setAttribute("disabled", "true");
      chatUI.showProgress(true);
      chatUI.enableChatInput(false);
      chatUI.appendMessage("warning", "Creating engine and downloading models...");
      
      await chatUI.initChatUI();
    };
    // TODO: find other alternative triggers
    getElementAndCheck("chat-user-input").onkeypress = (event) => {
      if (event.keyCode === 13) {
        chatUI.onGenerate();
      }
    };
    return chatUI;
  };

  private async initChatUI() {
    if (this.chatLoaded) return;
    this.requestInProgress = true;
    // Load models and create engine
    try {
      this.engine = await webllm.CreateMLCEngine(
        [this.embeddingModel, this.llmModel],
        {
          initProgressCallback: this.initProgressCallback,
          logLevel: "INFO", // specify the log level
        },
      );
      this.appendMessage("info", "Create vector store from product data. Standby...");
      // Process according to Snowflake model
      const knowledge: Document[] = values.map((item) => {
        return {pageContent: `[CLS] ${item.title} ${item.description}. price ${item.price}. rating ${item.rating}. discount ${item.discountPercentage} percent. [SEP]`, metadata: { id: item.id } };
      });
      const metadata = knowledge.map((_, index) => ({ id: index }));
      // Create document store
      const vectorStore = await MemoryVectorStore.fromDocuments(
        knowledge,
        new WebLLMEmbeddings(this.engine, this.embeddingModel),
      );
      this.updateLastMessage("info", " Vector store created. "+values.length+" JSON-documents added to the store.", false);
      //
      const retriever = vectorStore.asRetriever();
      // Create prompt
      const prompt =
      PromptTemplate.fromTemplate(`Answer the question based only on the following context:
      {context}
      
      Question: {question}`);
      // Create chain
      const chain = RunnableSequence.from([
        {
          context: retriever.pipe(formatDocumentsAsString),
          question: new RunnablePassthrough(),
        },
        prompt,
      ]);
      // When we detect low maxStorageBufferBindingSize, we assume that the device (e.g. an Android
      // phone) can only handle small models and make all other models unselectable. Otherwise, the
      // browser may crash. See https://github.com/mlc-ai/web-llm/issues/209.
      // Also use GPU vendor to decide whether it is a mobile device (hence with limited resources).
      const androidMaxStorageBufferBindingSize = 1 << 27; // 128MB
      const mobileVendors = new Set<string>(["qualcomm", "arm"]);
      let restrictModels = false;
      let maxStorageBufferBindingSize: number = 0;
      let gpuVendor: string = ""; // Initialize gpuVendor with a default value
      try {
        [maxStorageBufferBindingSize, gpuVendor] = await Promise.all([
          this.engine.getMaxStorageBufferBindingSize(),
          this.engine.getGPUVendor(),
        ]);
      } catch (err: Error | any) {
        this.appendMessage("danger", "Error reading GPU-vendor " + err.toString());
        console.log(err.stack);
      }
      if (
        (gpuVendor.length != 0 && mobileVendors.has(gpuVendor)) ||
        maxStorageBufferBindingSize <= androidMaxStorageBufferBindingSize
      ) {
        this.updateLastMessage(
          "info",
          "Your device seems to have limited resources, so we restrict the selectable models.",
          true
        );
        restrictModels = true;
      }
      // Update UI
      this.updateLastMessage("info", "\n\nChat assistant is ready", true);
      this.updateLastMessage("warning", "\n\nAll models loaded and engine initialized.", true);
      this.showProgress(false);
      this.enableChatInput(true);
      this.requestInProgress = false;
      this.chatLoaded = true;
    } catch (err: Error | any) {
      this.appendMessage("danger", err.toString());
      console.log(err.stack);
      this.uiChatDownloadButton.removeAttribute("disabled");
      this.showProgress(false);
      this.requestInProgress = false;
      this.chatLoaded = false;
    }
  }

  /**
   * Push a task to the execution queue.
   *
   * @param task The task to be executed;
   */
  private pushTask(task: () => Promise<void>) {
    const lastEvent = this.chatRequestChain;
    this.chatRequestChain = lastEvent.then(task);
  }

  // Event handlers
  // all event handler pushes the tasks to a queue
  // that get executed sequentially
  // the tasks previous tasks, which causes them to early stop
  // can be interrupted by engine.interruptGenerate
  private async onGenerate() {
    if (this.requestInProgress) {
      return;
    }
    this.pushTask(async () => {
      await this.asyncGenerate();
    });
  }

  private async onReset() {
    if (this.requestInProgress) {
      // interrupt previous generation if any
      this.engine.interruptGenerate();
    }
    // try reset after previous requests finishes
    this.pushTask(async () => {
      await this.engine.resetChat(true, this.llmModel); // TODO: Should keepStats be true or false?
      this.resetChatHistory();
    });
  }

  private async unloadEngine() {
    await this.engine.unload();
    this.chatLoaded = false;
    this.showProgress(false);
    this.enableChatInput(false);
    this.uiChatDownloadButton.removeAttribute("disabled");
    this.appendMessage("info", "Engine unloaded. You can download models again and retry.");
  }

  // Internal helper functions
  private async appendMessage(kind: string, text: string) {
    if (this.uiChat === undefined) {
      throw Error("cannot find ui chat");
    }
    let markdown = DOMPurify.sanitize(await marked.parse(text));
    const msg = `
      <div class="msg ${kind}-msg card bg-${kind}-subtle text-bg-${kind}-subtle my-2 p-2 ml-5">
        <div class="msg-bubble small">
          <div class="msg-text">${markdown}</div>
        </div>
      </div>
    `;
    this.uiChat.insertAdjacentHTML("beforeend", msg);
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);
  }

  // Special care for user input such that we treat it as pure text instead of html
  private async appendUserMessage(text: string) {
    if (this.uiChat === undefined) {
      throw Error("cannot find ui chat");
    }
    const msg = `
      <div class="msg primary-msg card bg-primary-subtle text-bg-primary-subtle my-2 p-2">
        <div class="msg-bubble small">
          <div class="msg-text"></div>
        </div>
      </div>
    `;
    this.uiChat.insertAdjacentHTML("beforeend", msg);
    // Recurse three times to get `msg-text`
    const msgElement = this.uiChat.lastElementChild?.lastElementChild
      ?.lastElementChild as HTMLElement;

    /*msgElement.insertAdjacentText("beforeend", text);
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);*/
    // Parse & sanitize markdown to HTML, then insert as HTML so it's rendered
    const markdown = DOMPurify.sanitize(await marked.parse(text));
    msgElement.insertAdjacentHTML("beforeend", markdown);
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);
  }

  private async updateLastMessage(kind: string, text: string, keepText: boolean = false) {
    if (this.uiChat === undefined) {
      throw Error("cannot find ui chat");
    }
    const matches = this.uiChat.getElementsByClassName(`msg ${kind}-msg`);
    if (matches.length == 0) throw Error(`${kind} message do not exist`);
    const msg = matches[matches.length - 1];
    const msgText = msg.getElementsByClassName("msg-text");
    if (msgText.length != 1) throw Error("Expect msg-text");
    
    // Parse the new markdown to sanitized HTML
    const parsed = DOMPurify.sanitize(await marked.parse(text));
    // If keepText is true, append the new parsed HTML to existing innerHTML; otherwise replace it
    const finalText = keepText ? msgText[0].innerHTML + parsed : parsed;
    
    if (msgText[0].innerHTML == finalText) return;
    // Set innerHTML so the sanitized HTML is rendered
    msgText[0].innerHTML = finalText;
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);
  }

  private resetChatHistory() {
    this.chatHistory = [];
    const clearTags = ["primary", "secondary", "warning", "error" , "danger", "info"];
    for (const tag of clearTags) {
      // need to unpack to list so the iterator don't get affected by mutation
      const matches = [...this.uiChat.getElementsByClassName(`msg ${tag}-msg`)];
      for (const item of matches) {
        this.uiChat.removeChild(item);
      }
    }
    if (this.uiChatInfoLabel !== undefined) {
      this.uiChatInfoLabel.innerHTML = "";
    }
    this.appendMessage("info", "Chat assistant is ready");
  }

  /**
   * Run generate
   */
  private async asyncGenerate() {
    //await this.asyncInitChat();
    this.requestInProgress = true;
    const prompt = this.uiChatInput.value;
    if (prompt == "") {
      this.requestInProgress = false;
      return;
    }
    this.appendUserMessage(prompt);
    this.uiChatInput.value = "";
    this.uiChatInput.setAttribute("placeholder", "Generating...");

    this.appendMessage("secondary", "... thinking ...");
    this.chatHistory.push({ role: "user", content: prompt });

    // Keep only the last 2 message pairs (4 messages) to maintain sliding window
    const maxMessages = 4;
    if (this.chatHistory.length > maxMessages) {
      this.chatHistory = this.chatHistory.slice(-maxMessages);
    }

    try {
      let curMessage = "";
      let usage: webllm.CompletionUsage | undefined = undefined;
      const completion = await this.engine.chat.completions.create({
        model: this.llmModel,
        stream: true,
        messages: this.chatHistory,
        stream_options: { include_usage: true },
        // if model starts with "Qwen3", disable thinking.
        extra_body: this.llmModel.startsWith("Qwen3")
          ? {
              enable_thinking: false,
            }
          : undefined,
      });
      // TODO(Charlie): Processing of � requires changes
      for await (const chunk of completion) {
        const curDelta = chunk.choices[0]?.delta.content;
        if (curDelta) {
          curMessage += curDelta;
        }
        this.updateLastMessage("secondary", curMessage);
        if (chunk.usage) {
          usage = chunk.usage;
        }
      }
      if (usage) {
        this.uiChatInfoLabel.innerHTML =
          `prompt: ${usage.prompt_tokens}, ` +
          `completion: ${usage.completion_tokens}, ` + "<br/>" +
          `prefill: ${usage.extra.prefill_tokens_per_s.toFixed(1)} tks/s, ` +
          `decoding: ${usage.extra.decode_tokens_per_s.toFixed(1)} tks/s`;
      }
      const finalMessage = await this.engine.getMessage(this.llmModel);
      this.updateLastMessage("secondary", finalMessage); // TODO: Remove this after � issue is fixed
      this.chatHistory.push({ role: "assistant", content: finalMessage });
    } catch (err: Error | any) {
      this.appendMessage("danger", "Generate error, " + err.toString());
      console.log(err.stack);
      await this.unloadEngine();
    }
    this.uiChatInput.setAttribute("placeholder", "Type a message...");
    this.requestInProgress = false;
  }
}

// For integration with Langchain
class WebLLMEmbeddings implements EmbeddingsInterface {
  engine: webllm.MLCEngineInterface;
  modelId: string;
  constructor(engine: webllm.MLCEngineInterface, modelId: string) {
    this.engine = engine;
    this.modelId = modelId;
  }

  async _embed(texts: string[]): Promise<number[][]> {
    const reply = await this.engine.embeddings.create({
      input: texts,
      model: this.modelId,
    });
    const result: number[][] = [];
    for (let i = 0; i < texts.length; i++) {
      result.push(reply.data[i].embedding);
    }
    return result;
  }

  async embedQuery(document: string): Promise<number[]> {
    return this._embed([document]).then((embeddings) => embeddings[0]);
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    return this._embed(documents);
  }
}

/* Configure marked.js library */
marked.use({
  async: true,
  gfm: true,
  breaks: true,
  pedantic: false,
});
/* Load the product list to List.js */
let options = {
  valueNames: [ 'title', 'description', 'price', 'rating', 'stock', 'discountPercentage', { name: 'thumbnail', attr: 'src' } ],
  item: '<li class="p-0 overflow-hidden my-2 mx-4"><div class="d-flex flex-row h-100 border border-1 small"><div class="p-3 flex-grow-1"><strong class="title mt-0 font-weight-bold mb-2">Title</strong><p class="description font-italic text-muted mb-0 small">Description</p><div class="small mt-1" style="display: grid;grid-template-columns: auto auto auto auto auto auto;gap: 4px 8px;"><div class="" style="display: flex; align-items: left;"><i class="bi bi-tag me-1"></i>Price:</div><div class="price"></div><div class="" style = "text-align: left;">$</div><div class="" style="display: flex; align-items: left;"><i class="bi bi-graph-up-arrow me-1"></i>Discount:</div><div class="discountPercentage"></div><div class="" style = "text-align: left;">%</div><div class="" style="display: flex; align-items: left;"><i class="bi bi-star me-1"></i>Rating:</div><div class="rating"></div><div class="" style = "text-align: left;"></div><div class="" style="display: flex; align-items: left;"><i class="bi bi-box-seam me-1"></i>Stock:</div><div class="stock"></div><div class="" style = "text-align: left;"></div></div></div><img class="thumbnail" alt="Product Thumbnail" style="width: 100px; object-fit: cover; min-height: 100%; margin-right: 8px;"/></div></li>'
};
let values = products.products;
const productList = new List('product-list', options, values);
// Create and initialize ChatUI
ChatUI.CreateAsync();