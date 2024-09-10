// Function to get Ollama model list
async function getOllamaModels() {
    const response = await fetch("http://localhost:11434/api/tags");
    if (!response.ok) {
        throw new Error("Failed to fetch Ollama models.");
    }
    const data = await response.json();
    return data.models.map(model => model.name);
}

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
function loadSettings() {
    return new Promise((resolve) => {
        chrome.storage.sync.get(chatSettings, (items) => {
            chatSettings = items;
            resolve();
        });
    });
}

// Keep track of the current tab and port
let currentTabId = null;
let currentPort = null;

// Handle connections from content script
chrome.runtime.onConnect.addListener(function(port) {
    if (port.name === "ollamaAI") {
        currentPort = port;
        currentTabId = port.sender.tab.id;

        port.onDisconnect.addListener(() => {
            currentPort = null;
            currentTabId = null;
        });

        port.onMessage.addListener(async function(request) {
            if (request.action === 'processText') {
                await loadSettings(); // Ensure settings are loaded before processing
                try {
                    const modelList = await getOllamaModels();
                    const apiKey = await getOpenAIKey();

                    const prompt = constructPrompt(request.text, chatSettings);
                    let apiEndpoint = "http://localhost:11434/api/generate"; // Default to Ollama
                    let payload = {
                        model: chatSettings.model,
                        prompt: prompt,
                        options: {
                            temperature: chatSettings.temperature,
                            max_tokens: chatSettings.max_tokens
                        },
                        stream: false
                    };

                    if (chatSettings.model.startsWith("gpt-")) {
                        apiEndpoint = "https://api.openai.com/v1/chat/completions";
                        payload = {
                            model: chatSettings.model,
                            messages: [{ role: 'user', content: prompt }],
                            temperature: chatSettings.temperature,
                            max_tokens: chatSettings.max_tokens
                        };
                    } else if ([
                        "llama-3.1-70b-versatile",
                        "llama-3.1-8b-instant",
                        "llama3-groq-70b-8192-tool-use-preview",
                        "llama3-groq-8b-8192-tool-use-preview",
                        "llama-guard-3-8b",
                        "llama3-70b-8192",
                        "llama3-8b-8192",
                        "mixtral-8x7b-32768",
                        "gemma-7b-it",
                        "gemma2-9b-it",
                    ].includes(chatSettings.model)) {
                        apiEndpoint = "https://api.groq.com/v1/chat/completions";
                        payload = {
                            model: chatSettings.model,
                            messages: [{ role: 'user', content: prompt }],
                            temperature: chatSettings.temperature,
                            max_tokens: chatSettings.max_tokens
                        };
                    }

                    const response = await fetch(apiEndpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            ...(chatSettings.model.startsWith("gpt-") && { 'Authorization': `Bearer ${apiKey}` }),
                            ...(!chatSettings.model.startsWith("gpt-") && chatSettings.model.includes("groq") && { 'Authorization': `Bearer ${apiKey}` })
                        },
                        body: JSON.stringify(payload)
                    });

                    if (!response.ok) {
                        throw new Error(`API request failed with status ${response.status}`);
                    }

                    const responseData = await response.json();
                    const output = chatSettings.model.startsWith("gpt-") || chatSettings.model.includes("groq") ? responseData.choices[0].message.content : responseData.response;
                    currentPort.postMessage({ output: output });
                } catch (error) {
                    console.error('Error in background script:', error);
                    currentPort.postMessage({ error: error.message });
                }
            }
        });
    }
});

// Fetch OpenAI API key from Ollama Workbench
async function getOpenAIKey() {
    try {
        const response = await fetch('http://localhost:8502/openai-key');
        if (!response.ok) {
            throw new Error(`Failed to fetch OpenAI API key: ${response.status}`);
        }
        const data = await response.json();
        return data.openai_api_key;
    } catch (error) {
        console.error('Error fetching OpenAI API key:', error);
        return null;
    }
}

function constructPrompt(text, settings) {
    let prompt = "";
    if (settings.agent_type !== "None") {
        prompt += getAgentPrompt(settings.agent_type) + "\n\n";
    }
    if (settings.metacognitive_type !== "None") {
        prompt += getMetacognitivePrompt(settings.metacognitive_type) + "\n\n";
    }
    if (settings.voice_type !== "None") {
        prompt += getVoicePrompt(settings.voice_type) + "\n\n";
    }
    prompt += `User: Please process the following text:\n\n${text}\n\nAssistant:`;
    return prompt;
}

function getAgentPrompt(agentType) {
    const prompts = {
        "Coder": "You are a highly skilled coder. You are able to write code in multiple programming languages. When responding to prompts, you format your responses as a code block.",
        "Analyst": "You are a professional data analyst. You are able to perform in-depth analysis and extract meaningful insights from given datasets.",
        "Creative Writer": "You are a talented creative writer. Your writing style is imaginative, engaging, and tailored to captivate your audience."
    };
    return prompts[agentType] || "You are a helpful AI assistant.";
}

function getMetacognitivePrompt(metacognitiveType) {
    const prompts = {
        "Visualization of Thought": "Before responding, visualize the problem and the steps needed to solve it. Describe your thought process clearly, including the visualization.",
        "Chain of Thought": "Break down the problem into smaller steps and reason through each step carefully. Explain your reasoning for each step in detail.",
    };
    return prompts[metacognitiveType] || "";
}

function getVoicePrompt(voiceType) {
    const prompts = {
        "Friendly": "Use a casual and approachable tone. You aim to be helpful and understanding in your responses.",
        "Professional": "Maintain a formal and concise tone. Your responses should be informative and objective.",
        "Humorous": "Use humor and wit in your responses. You aim to entertain and engage the user."
    };
    return prompts[voiceType] || "";
}

// Handle messages from popup.js (for setting changes)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'updateSettings') {
        chatSettings = request.settings;
        console.log('Settings updated in background:', chatSettings);
        sendResponse({ success: true });
    }
    return true;  // Indicate that the response is sent asynchronously
});

// Load settings when the background script starts
loadSettings();