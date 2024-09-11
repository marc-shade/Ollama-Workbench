// background.js

// Default chat settings
let chatSettings = {
    model: 'mistral:instruct',
    temperature: 0.7,
    max_tokens: 1000,
    agent_type: "None",
    metacognitive_type: "None",
    voice_type: "None"
  };
  
  // Load settings from storage or use defaults
  chrome.storage.sync.get(chatSettings, (items) => {
    chatSettings = items;
  });
  
  // Listen for a message from main.py containing the port number
  chrome.runtime.onConnectExternal.addListener(function(port) {
    console.log("Connected to native host", port);
    port.onMessage.addListener(function(msg) {
      console.log("Received message:", msg);
      if ('port' in msg) {
        localStorage.setItem("ollamaPort", msg.port);
        console.log("Ollama port received:", msg.port);
      }
    });
  });
  
  
  // Handle messages from popup.js (for setting changes)
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'updateSettings') {
      chatSettings = request.settings;
      chrome.storage.sync.set(chatSettings, () => {
        console.log('Settings updated:', chatSettings);
        sendResponse({ success: true });
      });
      return true; // Indicate asynchronous response
    }
  });

  chrome.runtime.onInstalled.addListener(() => {
    console.log('Ollama Workbench Extension installed');
  });
  