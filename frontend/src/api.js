const BASE = "/api";

async function readNdjson(response, onEvent) {
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      // ignore
    }
    throw new Error(message);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) {
    onEvent(JSON.parse(buffer));
  }
}

export async function listConversations() {
  const res = await fetch(`${BASE}/conversations`);
  return res.json();
}

export async function newConversation() {
  const res = await fetch(`${BASE}/conversations`, { method: "POST" });
  const data = await res.json();
  return data.id;
}

export async function getConversation(id) {
  const res = await fetch(`${BASE}/conversations/${id}`);
  const data = await res.json();
  return data.messages;
}

export async function deleteConversation(id) {
  await fetch(`${BASE}/conversations/${id}`, { method: "DELETE" });
}

export async function getKnowledgeBase() {
  const res = await fetch(`${BASE}/knowledge-base`);
  return res.json();
}

export async function clearKnowledgeBase() {
  await fetch(`${BASE}/knowledge-base`, { method: "DELETE" });
}

export async function deleteSource(source) {
  const res = await fetch(`${BASE}/knowledge-base/${encodeURIComponent(source)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `删除失败（${res.status}）`);
  }
  return res.json();
}

export async function uploadFiles(files, onEvent) {
  const formData = new FormData();
  for (const file of files) formData.append("files", file);
  const response = await fetch(`${BASE}/upload`, { method: "POST", body: formData });
  await readNdjson(response, onEvent);
}

export async function chat(conversationId, question, onEvent) {
  const response = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, question }),
  });
  await readNdjson(response, onEvent);
}
