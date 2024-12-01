// background.js

// Default chat settings
let chatSettings = {
    model: 'llama2',  // Changed from mistral:instruct to a more commonly available model
    temperature: 0.7,
    max_tokens: 1000,
    agent_type: "None",
    metacognitive_type: "None",
    voice_type: "None"
};

// Store the current port
let currentPort = null;

// Load settings from storage or use defaults
chrome.storage.sync.get(chatSettings, (items) => {
    chatSettings = items;
});

// Function to check if a port is available
async function checkPort(port) {
    try {
        console.log(`Checking port ${port}...`);
        const response = await fetch(`http://127.0.0.1:${port}/port`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            mode: 'cors',
            credentials: 'include'
        });
        
        if (response.ok) {
            const actualPort = await response.json();
            console.log(`Server confirmed on port ${actualPort}`);
            return actualPort;
        }
        console.log(`Port ${port} check failed with status: ${response.status}`);
    } catch (error) {
        console.log(`Error checking port ${port}:`, error);
    }
    return null;
}

// Function to discover the correct port
async function discoverPort() {
    console.log('Starting port discovery...');
    // Try Streamlit port first, then others
    const portsToTry = [8501, 8502, 8503, 8504, 8505];
    
    for (const port of portsToTry) {
        console.log(`Checking port ${port}...`);
        const actualPort = await checkPort(port);
        if (actualPort) {
            console.log(`Found server on port ${actualPort}`);
            return actualPort;
        }
        await new Promise(resolve => setTimeout(resolve, 500)); // Small delay between checks
    }
    
    console.log('No server found on any port');
    return null;
}

// Function to handle context menu clicks
async function handleContextClick(info, tab) {
    console.log('Context menu clicked, info:', info);
    
    try {
        // Try to discover the port if we don't have one
        if (!currentPort) {
            currentPort = await discoverPort();
            if (!currentPort) {
                console.error('Could not find Ollama Workbench server');
                return;
            }
        }
        
        // Create the popup URL with the selected text
        let popupUrl = chrome.runtime.getURL('popup.html');
        if (info.selectionText) {
            popupUrl += `?text=${encodeURIComponent(info.selectionText)}`;
        }
        
        // Open the popup
        chrome.windows.create({
            url: popupUrl,
            type: 'popup',
            width: 400,
            height: 600
        });
    } catch (error) {
        console.error('Error handling context menu click:', error);
    }
}

// Create context menu items
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: 'ollamaWorkbench',
        title: 'Open in Ollama Workbench',
        contexts: ['selection', 'page']
    });
});

// Listen for context menu clicks
chrome.contextMenus.onClicked.addListener(handleContextClick);

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Received message:', message);
    
    if (message.type === 'getPort') {
        if (currentPort) {
            console.log(`Returning cached port: ${currentPort}`);
            sendResponse({ port: currentPort });
        } else {
            console.log('No cached port, starting discovery...');
            discoverPort().then(port => {
                currentPort = port;
                console.log(`Discovered port: ${port}`);
                sendResponse({ port });
            }).catch(error => {
                console.error('Port discovery failed:', error);
                sendResponse({ error: 'Could not find server' });
            });
        }
        return true; // Will respond asynchronously
    } else if (message.action === 'updateSettings') {
        chatSettings = message.settings;
        chrome.storage.sync.set(chatSettings, () => {
            console.log('Settings updated:', chatSettings);
            sendResponse({ success: true });
        });
        return true; // Indicate asynchronous response
    }
});