<script setup>
import { ref } from "vue";
import * as api from "../api";

const props = defineProps({
  conversations: { type: Array, required: true },
  activeId: { type: String, default: null },
  kb: { type: Object, required: true },
});
const emit = defineEmits(["new-conversation", "switch-conversation", "delete-conversation", "kb-changed"]);

const selectedFiles = ref([]);
const fileInput = ref(null);
const uploading = ref(false);
const uploadStatus = ref("");
const uploadError = ref("");
const clearing = ref(false);

function onFilesPicked(e) {
  selectedFiles.value = Array.from(e.target.files || []);
}

async function processDocuments() {
  if (!selectedFiles.value.length || uploading.value) return;
  uploading.value = true;
  uploadError.value = "";
  uploadStatus.value = "开始处理...";
  try {
    await api.uploadFiles(selectedFiles.value, (event) => {
      if (event.type === "page_progress") {
        uploadStatus.value = `正在转录《${event.file}》：${event.done}/${event.total} 页`;
      } else if (event.type === "file_done") {
        uploadStatus.value = `已完成《${event.file}》（${event.pages} 页），准备生成片段上下文…`;
      } else if (event.type === "replaced") {
        uploadStatus.value = `《${event.file}》已在知识库中，替换原有的 ${event.removed} 个片段…`;
      } else if (event.type === "context_progress") {
        uploadStatus.value = `正在生成片段上下文：${event.done}/${event.total}`;
      } else if (event.type === "image_index_progress") {
        uploadStatus.value = `正在建立图像检索索引：${event.done}/${event.total}`;
      } else if (event.type === "error") {
        uploadError.value = event.message;
      } else if (event.type === "complete") {
        uploadStatus.value = `处理完成，知识库现有 ${event.size} 个片段`;
        emit("kb-changed");
      }
    });
  } catch (e) {
    uploadError.value = e.message || "处理失败";
  } finally {
    uploading.value = false;
    selectedFiles.value = [];
    if (fileInput.value) fileInput.value.value = "";
  }
}

async function clearKnowledgeBase() {
  if (!confirm("确定要清空整个知识库吗？所有手册都需要重新上传处理。")) return;
  clearing.value = true;
  try {
    await api.clearKnowledgeBase();
    emit("kb-changed");
  } finally {
    clearing.value = false;
  }
}

async function removeSource(source) {
  if (!confirm(`确定要从知识库中删除《${source}》吗？`)) return;
  clearing.value = true;
  uploadError.value = "";
  try {
    await api.deleteSource(source);
    emit("kb-changed");
  } catch (e) {
    uploadError.value = e.message || "删除失败";
  } finally {
    clearing.value = false;
  }
}
</script>

<template>
  <aside class="sidebar">
    <h3>💬 对话历史</h3>
    <button class="btn btn-primary btn-block" @click="$emit('new-conversation')">＋ 新对话</button>

    <div class="conversation-list">
      <div
        v-for="conv in conversations"
        :key="conv.id"
        class="conversation-row"
        :class="{ active: conv.id === activeId }"
      >
        <button class="conv-title" @click="$emit('switch-conversation', conv.id)">
          {{ conv.title }}
        </button>
        <button class="conv-delete" title="删除对话" @click="$emit('delete-conversation', conv.id)">🗑</button>
      </div>
    </div>

    <h3>📄 上传手册 PDF</h3>
    <input ref="fileInput" type="file" accept=".pdf" multiple @change="onFilesPicked" />
    <button
      class="btn btn-primary btn-block"
      style="margin-top: 8px"
      :disabled="!selectedFiles.length || uploading"
      @click="processDocuments"
    >
      {{ uploading ? "处理中…" : "⚡ 处理文档" }}
    </button>
    <p v-if="uploadStatus" class="hint">{{ uploadStatus }}</p>
    <p v-if="uploadError" class="hint error">{{ uploadError }}</p>

    <template v-if="kb.size">
      <p class="hint">当前知识库：{{ kb.size }} 个片段，共 {{ kb.sources.length }} 份手册</p>
      <div class="source-list">
        <div v-for="source in kb.sources" :key="source" class="source-row">
          <span class="source-name" :title="source">{{ source }}</span>
          <button
            class="conv-delete"
            :disabled="clearing"
            title="从知识库中删除这份手册"
            @click="removeSource(source)"
          >
            🗑
          </button>
        </div>
      </div>
      <button class="btn" :disabled="clearing" @click="clearKnowledgeBase">🗑 清空知识库</button>
    </template>
  </aside>
</template>

<style scoped>
/* 侧栏用半透明材质，内容在其下方滚动时会透出来 */
.sidebar {
  width: 272px;
  flex-shrink: 0;
  background: var(--bg-sidebar);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-right: 1px solid var(--separator);
  padding: 16px 12px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sidebar h3 {
  padding: 0 8px;
  margin: 14px 0 2px;
}

.sidebar h3:first-child {
  margin-top: 0;
}

/* —— 对话列表 —— */
.conversation-list {
  display: flex;
  flex-direction: column;
  gap: 1px;
  margin-bottom: 4px;
}

.conversation-row {
  display: flex;
  align-items: center;
  border-radius: var(--r-sm);
  transition: background 0.16s ease;
}

.conversation-row:hover {
  background: var(--bg-hover);
}

.conversation-row.active {
  background: var(--bg-active);
}

.conv-title {
  flex: 1;
  min-width: 0;
  text-align: left;
  border: none;
  background: none;
  color: var(--label);
  font-size: 15px;
  letter-spacing: -0.01em;
  padding: 7px 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.conversation-row.active .conv-title {
  font-weight: 590;
  color: var(--blue);
}

/* 删除按钮平时隐藏，悬停才出现——减少视觉噪声 */
.conv-delete {
  border: none;
  background: none;
  color: var(--label-3);
  font-size: 13px;
  padding: 6px 9px;
  border-radius: var(--r-sm);
  opacity: 0;
  transition: opacity 0.16s ease, color 0.16s ease;
}

.conversation-row:hover .conv-delete,
.source-row:hover .conv-delete,
.conv-delete:focus-visible {
  opacity: 1;
}

.conv-delete:hover:not(:disabled) {
  color: var(--red);
}

/* —— 上传 —— */
.sidebar input[type="file"] {
  font-size: 13px;
  color: var(--label-2);
  padding: 0 8px;
  max-width: 100%;
}

.sidebar input[type="file"]::file-selector-button {
  font-family: inherit;
  font-size: 13px;
  font-weight: 510;
  border: none;
  background: var(--bg-input);
  color: var(--label);
  border-radius: var(--r-full);
  padding: 5px 13px;
  margin-right: 8px;
  cursor: pointer;
}

.hint {
  font-size: 13px;
  line-height: 1.4;
  color: var(--label-2);
  padding: 0 8px;
  margin: 2px 0;
}

.hint.error {
  color: var(--red);
}

/* —— 知识库文档列表 —— */
.source-list {
  display: flex;
  flex-direction: column;
  gap: 1px;
  margin-bottom: 4px;
}

.source-row {
  display: flex;
  align-items: center;
  border-radius: var(--r-sm);
  transition: background 0.16s ease;
}

.source-row:hover {
  background: var(--bg-hover);
}

.source-name {
  flex: 1;
  min-width: 0;
  font-size: 13px;
  color: var(--label-2);
  padding: 6px 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

@media (max-width: 720px) {
  .sidebar { width: 216px; }
}
</style>
