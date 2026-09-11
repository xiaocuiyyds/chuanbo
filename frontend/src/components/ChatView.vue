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
.chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 0 24px;
  height: 100%;
}

.app-title {
  padding-top: 20px;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px 0 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.message {
  display: flex;
  gap: 12px;
  max-width: 900px;
}

.avatar {
  flex-shrink: 0;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--bg-sidebar-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
}

.bubble {
  flex: 1;
  min-width: 0;
}

.sources {
  margin-top: 10px;
}

.sources-toggle {
  font-size: 13px;
}

.sources-body {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.source-item {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  background: var(--bg-card);
}

.source-head {
  font-size: 13px;
  margin-bottom: 6px;
}

.text-muted {
  color: var(--text-muted);
}

.source-text {
  white-space: pre-wrap;
  font-family: inherit;
  font-size: 13px;
  color: var(--text-muted);
  margin: 0 0 8px;
}

.source-image {
  max-width: 100%;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}

.error-banner {
  color: var(--primary-dark);
  background: var(--bg-sidebar-secondary);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  font-size: 14px;
}

.composer {
  display: flex;
  gap: 8px;
  padding: 12px 0 20px;
  border-top: 1px solid var(--border);
}

.composer input {
  flex: 1;
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text);
  border-radius: var(--radius);
  padding: 12px 16px;
  font-size: 14px;
  font-family: inherit;
}

.composer input:focus {
  outline: 2px solid var(--primary);
  outline-offset: -1px;
}

.composer button {
  width: 44px;
  border-radius: var(--radius);
  font-size: 16px;
}
</style>
