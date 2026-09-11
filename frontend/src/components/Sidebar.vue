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
.sidebar {
  width: 280px;
  flex-shrink: 0;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  padding: 20px 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.source-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 6px;
}

.source-row {
  display: flex;
  align-items: center;
  gap: 4px;
}

.source-name {
  flex: 1;
  min-width: 0;
  font-size: 12px;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.conversation-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 12px;
}

.conversation-row {
  display: flex;
  align-items: center;
  gap: 4px;
}

.conv-title {
  flex: 1;
  text-align: left;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  padding: 8px 10px;
  color: var(--text);
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conv-title:hover {
  background: var(--bg-sidebar-secondary);
}

.conversation-row.active .conv-title {
  background: var(--primary);
  color: #fff;
  font-weight: 600;
}

.conv-delete {
  background: transparent;
  border: none;
  padding: 6px;
  border-radius: var(--radius-sm);
  font-size: 12px;
}

.conv-delete:hover {
  background: var(--bg-sidebar-secondary);
}

input[type="file"] {
  width: 100%;
  font-size: 12px;
  color: var(--text-muted);
}

.hint {
  font-size: 12px;
  color: var(--text-muted);
  margin: 8px 0;
}

.hint.error {
  color: var(--primary-dark);
}
</style>
