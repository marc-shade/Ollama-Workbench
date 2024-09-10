// Function to populate select dropdowns
async function populateDropdowns() {
    const modelSelect = document.getElementById('modelSelect');
    const agentTypeSelect = document.getElementById('agentTypeSelect');
    const metacognitiveTypeSelect = document.getElementById('metacognitiveTypeSelect');
    const voiceTypeSelect = document.getElementById('voiceTypeSelect');

    try {
        const response = await fetch('http://localhost:8503/prompts');
        if (!response.ok) {
            throw new Error(`Failed to fetch prompts from Ollama Workbench: ${response.status}`);
        }
        const prompts = await response.json();

        // Populate model select
        const modelResponse = await fetch('http://localhost:11434/api/tags');
        if (!modelResponse.ok) {
            throw new Error(`Failed to fetch models from Ollama: ${modelResponse.status}`);
        }
        const modelData = await modelResponse.json();
        modelData.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.text = model.name;
            modelSelect.add(option);
        });

        // Populate other selects
        populateSelect(agentTypeSelect, prompts.agent, "Agent");
        populateSelect(metacognitiveTypeSelect, prompts.metacognitive, "Metacognitive");
        populateSelect(voiceTypeSelect, prompts.voice, "Voice");

        // Load saved settings after populating dropdowns
        loadSettings();

    } catch (error) {
        console.error('Error populating dropdowns:', error);
        const errorMessage = document.createElement('p');
        errorMessage.textContent = `Error: ${error.message}`;
        errorMessage.style.color = 'red';
        document.body.appendChild(errorMessage);
    }
}

// Helper function to populate a select element
function populateSelect(selectElement, options, label) {
    const noneOption = document.createElement('option');
    noneOption.value = "None";
    noneOption.text = `None (${label})`;
    selectElement.add(noneOption);

    for (const key in options) {
        const option = document.createElement('option');
        option.value = key;
        option.text = key;
        selectElement.add(option);
    }
}

// Function to save settings automatically
function saveSettings() {
    const settings = {
        model: document.getElementById('modelSelect').value,
        agent_type: document.getElementById('agentTypeSelect').value,
        metacognitive_type: document.getElementById('metacognitiveTypeSelect').value,
        voice_type: document.getElementById('voiceTypeSelect').value,
        temperature: parseFloat(document.getElementById('temperatureInput').value),
        max_tokens: parseInt(document.getElementById('maxTokensInput').value)
    };

    chrome.storage.sync.set(settings, () => {
        if (chrome.runtime.lastError) {
            console.error("Error saving settings:", chrome.runtime.lastError);
        } else {
            console.log('Settings saved automatically');
            chrome.runtime.sendMessage({ action: 'updateSettings', settings: settings });
        }
    });
}

// Load settings from storage and update UI
function loadSettings() {
    chrome.storage.sync.get({
        model: 'mistral:instruct',
        agent_type: "None",
        metacognitive_type: "None",
        voice_type: "None",
        temperature: 0.7,
        max_tokens: 1000
    }, (items) => {
        document.getElementById('modelSelect').value = items.model;
        document.getElementById('agentTypeSelect').value = items.agent_type;
        document.getElementById('metacognitiveTypeSelect').value = items.metacognitive_type;
        document.getElementById('voiceTypeSelect').value = items.voice_type;
        document.getElementById('temperatureInput').value = items.temperature;
        document.getElementById('maxTokensInput').value = items.max_tokens;
    });
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    populateDropdowns();
    
    document.getElementById('toggleSidebar').addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, { action: 'toggleSidebar' });
        });
    });

    // Add event listeners for all input elements
    const inputElements = document.querySelectorAll('select, input');
    inputElements.forEach(element => {
        element.addEventListener('change', saveSettings);
    });
});