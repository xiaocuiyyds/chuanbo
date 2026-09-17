<script setup>
import { computed, nextTick, ref, watch } from "vue";
import { marked } from "marked";
import * as api from "../api";

const props = defineProps({
  conversationId: { type: String, default: null },
  messages: { type: Array, required: true },
  kbReady: { type: Boolean, default: false },
});
const emit = defineEmits(["answer-saved"]);

const localMessages = ref([]);
const question = ref("");
const isStreaming = ref(false);
const streamingText = ref("");
const streamingSources = ref(null);
const errorMessage = ref("");
const scrollEl = ref(null);
const expandedSources = ref({});

watch(
  () => [props.conversationId, props.messages],
  () => {
    localMessages.value = props.messages.map((m, i) => ({ ...m, _key: `${props.conversationId}-${i}` }));
    isStreaming.value = false;
    streamingText.value = "";
    streamingSources.value = null;
    errorMessage.value = "";
    scrollToBottom();
  },
  { immediate: true, deep: false }
);

function scrollToBottom() {
  nextTick(() => {
    if (scrollEl.value) scrollEl.value.scrollTop = scrollEl.value.scrollHeight;
  });
}

function renderMarkdown(text) {
  return marked.parse(text || "");
}

function toggleSources(key) {
  expandedSources.value[key] = !expandedSources.value[key];
}

async function submitQuestion() {
  const q = question.value.trim();
  if (!q || isStreaming.value || !props.conversationId) return;
  if (!props.kbReady) {
    errorMessage.value = "请先在侧边栏上传并处理 PDF 文档";
    return;
  }

  errorMessage.value = "";
  const userMsg = { role: "user", content: q, _key: `local-user-${Date.now()}` };
  localMessages.value.push(userMsg);
  question.value = "";
  isStreaming.value = true;
  streamingText.value = "";
  streamingSources.value = null;
  scrollToBottom();

  try {
    await api.chat(props.conversationId, q, (event) => {
      if (event.type === "sources") {
        streamingSources.value = event.sources;
      } else if (event.type === "delta") {
        streamingText.value += event.text;
        scrollToBottom();
      } else if (event.type === "error") {
        errorMessage.value = event.message;
      } else if (event.type === "done") {
        localMessages.value.push({
          role: "assistant",
          content: streamingText.value,
          sources: streamingSources.value || [],
          _key: `local-assistant-${Date.now()}`,
        });
        streamingText.value = "";
        streamingSources.value = null;
        emit("answer-saved");
      }
    });
  } catch (e) {
    errorMessage.value = e.message || "请求失败";
  } finally {
    isStreaming.value = false;
    scrollToBottom();
  }
}
</script>

<template>
  <main class="chat">
    <h1 class="app-title">🚢 船舶操作知识问答</h1>

    <div class="messages" ref="scrollEl">
      <div v-for="msg in localMessages" :key="msg._key" class="message" :class="msg.role">
        <div class="avatar">{{ msg.role === "user" ? "🧑" : "🤖" }}</div>
        <div class="bubble">
          <div class="markdown" v-html="renderMarkdown(msg.content)"></div>
          <div v-if="msg.sources && msg.sources.length" class="sources">
            <button class="btn sources-toggle" @click="toggleSources(msg._key)">
              📄 参考来源 {{ expandedSources[msg._key] ? "▲" : "▼" }}
            </button>
            <div v-if="expandedSources[msg._key]" class="sources-body">
              <div v-for="(src, i) in msg.sources" :key="i" class="source-item">
                <div class="source-head">
                  <strong>{{ src.source }} 第{{ src.page }}页</strong>
                  <span class="text-muted">（相关度 {{ src.score.toFixed(3) }}）</span>
                </div>
                <pre class="source-text">{{ src.text }}</pre>
                <img v-if="src.image_path" :src="`/images/${src.image_path}`" class="source-image" />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-if="isStreaming" class="message assistant">
        <div class="avatar">🤖</div>
        <div class="bubble">
          <div class="markdown" v-html="renderMarkdown(streamingText + '▌')"></div>
        </div>
      </div>

      <p v-if="errorMessage" class="error-banner">{{ errorMessage }}</p>
    </div>

    <form class="composer" @submit.prevent="submitQuestion">
      <input
        v-model="question"
        type="text"
        placeholder="请输入你的问题，例如：主机遥控出现XX报警应该怎么处理？"
        :disabled="isStreaming"
      />
      <button class="btn btn-primary" type="submit" :disabled="isStreaming || !question.trim()">↑</button>
    </form>
  </main>
</template>

<style scoped>
/* 布局：内容居中、宽度受限，让长回答保持可读的行长 */
.chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  height: 100%;
  position: relative;
}

.app-title {
  font-size: 17px;
  font-weight: 590;
  letter-spacing: -0.01em;
  text-align: center;
  padding: 14px 24px;
  margin: 0;
  flex-shrink: 0;
  /* 顶栏用半透明材质，内容从下方滑过时透出来 */
  background: var(--bg-sidebar);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid var(--separator);
  z-index: 2;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px 24px 120px;   /* 底部留出悬浮输入框的空间 */
  display: flex;
  flex-direction: column;
  gap: 22px;
  scroll-behavior: smooth;
}

.message {
  display: flex;
  gap: 12px;
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  animation: rise 0.28s cubic-bezier(0.22, 1, 0.36, 1);
}

@keyframes rise {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: none; }
}

/* 用户消息靠右，气泡用强调色——iMessage 的做法 */
.message.user {
  flex-direction: row-reverse;
}

.avatar {
  flex-shrink: 0;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: var(--bg-input);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  margin-top: 2px;
}

.message.assistant .avatar {
  background: var(--blue);
}

.bubble {
  min-width: 0;
  max-width: 88%;
}

.message.user .bubble {
  background: var(--blue);
  color: #fff;
  padding: 10px 16px;
  border-radius: var(--r-lg);
  max-width: 72%;
}

.message.user :deep(.markdown p) {
  margin: 0;
}

.message.assistant .bubble {
  background: var(--bg-elevated);
  padding: 14px 18px;
  border-radius: var(--r-lg);
  box-shadow: var(--shadow-sm);
}

/* —— 参考来源：默认收起，不干扰阅读 —— */
.sources {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--separator);
}

.sources-toggle {
  font-size: 13px;
  padding: 4px 12px;
  background: transparent;
  color: var(--blue);
  font-weight: 510;
}

.sources-toggle:hover:not(:disabled) {
  background: var(--bg-hover);
}

.sources-body {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.source-item {
  background: var(--bg-input);
  border-radius: var(--r-md);
  padding: 12px 14px;
}

.source-head {
  font-size: 13px;
  margin-bottom: 7px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: baseline;
}

.source-head strong {
  font-weight: 590;
}

.text-muted {
  color: var(--label-2);
  font-variant-numeric: tabular-nums;
}

.source-text {
  white-space: pre-wrap;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
  color: var(--label-2);
  margin: 0 0 10px;
  max-height: 180px;
  overflow-y: auto;
}

.source-image {
  max-width: 100%;
  border-radius: var(--r-sm);
  display: block;
}

.error-banner {
  max-width: 760px;
  margin: 0 auto;
  width: 100%;
  color: var(--red);
  background: color-mix(in srgb, var(--red) 10%, transparent);
  border-radius: var(--r-md);
  padding: 12px 16px;
  font-size: 15px;
}

/* —— 输入框：悬浮的毛玻璃胶囊 —— */
.composer {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  gap: 10px;
  align-items: center;
  /* 输入区整体居中，与上方消息列同宽，而不是把两端撑开 */
  width: min(760px, 100%);
  margin: 0 auto;
  padding: 16px 24px 22px;
  background: linear-gradient(to top, var(--bg) 62%, transparent);
  z-index: 2;
}

.composer input {
  flex: 1;
  min-width: 0;
  border: 1px solid var(--separator);
  background: var(--bg-elevated);
  color: var(--label);
  border-radius: var(--r-full);
  padding: 12px 20px;
  font-size: 17px;
  font-family: inherit;
  letter-spacing: -0.012em;
  box-shadow: var(--shadow-md);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  transition: border-color 0.18s ease;
}

.composer input::placeholder {
  color: var(--label-3);
}

.composer input:focus {
  outline: none;
  border-color: var(--blue);
}

.composer button {
  width: 44px;
  height: 44px;
  padding: 0;
  border-radius: 50%;
  font-size: 19px;
  font-weight: 700;
  box-shadow: var(--shadow-md);
  flex-shrink: 0;
}

@media (max-width: 720px) {
  .messages { padding: 16px 16px 112px; }
  .composer { padding: 12px 16px 18px; }
  .bubble, .message.user .bubble { max-width: 100%; }
}
</style>
