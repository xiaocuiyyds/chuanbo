<script setup>
import { onMounted, reactive, ref } from "vue";
import Sidebar from "./components/Sidebar.vue";
import ChatView from "./components/ChatView.vue";
import * as api from "./api";

const conversationId = ref(null);
const messages = ref([]);
const conversations = ref([]);
const kb = reactive({ size: 0, sources: [] });

async function refreshConversations() {
  conversations.value = await api.listConversations();
}

async function refreshKb() {
  const data = await api.getKnowledgeBase();
  kb.size = data.size;
  kb.sources = data.sources;
}

async function startNewConversation() {
  conversationId.value = await api.newConversation();
  messages.value = [];
}

async function switchConversation(id) {
  conversationId.value = id;
  messages.value = await api.getConversation(id);
}

async function removeConversation(id) {
  await api.deleteConversation(id);
  if (conversationId.value === id) {
    await startNewConversation();
  }
  await refreshConversations();
}

async function afterAnswerSaved() {
  await refreshConversations();
}

onMounted(async () => {
  await Promise.all([refreshConversations(), refreshKb()]);
  await startNewConversation();
});
</script>

<template>
  <div class="layout">
    <Sidebar
      :conversations="conversations"
      :active-id="conversationId"
      :kb="kb"
      @new-conversation="startNewConversation"
      @switch-conversation="switchConversation"
      @delete-conversation="removeConversation"
      @kb-changed="refreshKb"
    />
    <ChatView
      :conversation-id="conversationId"
      :messages="messages"
      :kb-ready="kb.size > 0"
      @answer-saved="afterAnswerSaved"
    />
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  height: 100%;
}
</style>
