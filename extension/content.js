document.addEventListener('DOMContentLoaded', function() {
    // Global variable to store the sidebar iframe
    let sidebarIframe = null;

    // Context Menu Setup
    chrome.runtime.onInstalled.addListener(() => {
        chrome.contextMenus.create({
            id: "ollamaAI",
            title: "Ollama AI Assistant",
            contexts: ["selection", "page"]
        });
    });

    // Handle context menu click
    chrome.contextMenus.onClicked.addListener((info, tab) => {
        if (info.menuItemId === "ollamaAI") {
            if (info.selectionText) {
                handleAIRequest(info.selectionText);
            } else {
                handleAIRequest(getPageContext());
            }
        }
    });

    // Function to get entire page text
    function getPageContext() {
        return document.body.innerText;
    }

    // Function to send text to background script for AI processing
    function handleAIRequest(text) {
        console.log("Processing Request...");

        // Establish a long-lived connection to the background script
        const port = chrome.runtime.connect({name: "ollamaAI"});

        // Send the text to the background script
        port.postMessage({ action: 'processText', text: text });

        // Listen for the response from the background script
        port.onMessage.addListener(function(response) {
            if (response.error) {
                alert(`Ollama AI Assistant Error: ${response.error}`);
            } else {
                showOutputOverlay(response.output);
            }

            // Disconnect the port after receiving the response
            port.disconnect();
        });

        return true;
    }

    // Create and display the overlay
    function showOutputOverlay(outputText) {
        let overlay = document.createElement('div');
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '100%';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        overlay.style.display = 'flex';
        overlay.style.justifyContent = 'center';
        overlay.style.alignItems = 'center';
        overlay.style.zIndex = '10000';

        let outputBox = document.createElement('div');
        outputBox.style.backgroundColor = '#fff';
        outputBox.style.padding = '20px';
        outputBox.style.borderRadius = '5px';
        outputBox.style.maxWidth = '80%';
        outputBox.style.maxHeight = '80%';
        outputBox.style.overflowY = 'auto';
        outputBox.style.textAlign = 'left';
        outputBox.innerText = outputText;

        // Close button
        let closeButton = document.createElement('button');
        closeButton.innerText = 'Close';
        closeButton.style.marginTop = '10px';
        closeButton.style.padding = '5px 10px';
        closeButton.addEventListener('click', () => {
            document.body.removeChild(overlay);
        });

        // Save to text file
        let saveToTextButton = document.createElement('button');
        saveToTextButton.innerText = 'Save to Text File';
        saveToTextButton.style.marginTop = '10px';
        saveToTextButton.style.marginLeft = '10px';
        saveToTextButton.style.padding = '5px 10px';
        saveToTextButton.addEventListener('click', () => {
            let blob = new Blob([outputText], { type: "text/plain;charset=utf-8" });
            let url = URL.createObjectURL(blob);

            let link = document.createElement('a');
            link.href = url;
            link.download = "ollama_output.txt";

            document.body.appendChild(link);
            link.click();

            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        });

        // Paste to Bootstrap HTML
        let saveToBootstrapHTMLButton = document.createElement('button');
        saveToBootstrapHTMLButton.innerText = 'Paste to Bootstrap HTML';
        saveToBootstrapHTMLButton.style.marginTop = '10px';
        saveToBootstrapHTMLButton.style.marginLeft = '10px';
        saveToBootstrapHTMLButton.style.padding = '5px 10px';
        saveToBootstrapHTMLButton.addEventListener('click', () => {
            copyToClipboard(outputText);
            window.open('file:///Volumes/FILES/code/edit_pad/edit.html', '_blank');
        });

        // Add elements to the DOM
        outputBox.appendChild(closeButton);
        outputBox.appendChild(saveToTextButton);
        outputBox.appendChild(saveToBootstrapHTMLButton);
        overlay.appendChild(outputBox);
        document.body.appendChild(overlay);
    }

    // Copy to clipboard function
    function copyToClipboard(text) {
        const el = document.createElement('textarea');
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
    }

    // Handle messages from popup
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === 'toggleSidebar') {
            toggleAISidebar();
            sendResponse({ success: true });
        }
        return true;  // Indicate asynchronous response
    });

    // Sidebar toggle function 
    function toggleAISidebar() {
        if (sidebarIframe) {
            sidebarIframe.remove();
            sidebarIframe = null;
        } else {
            sidebarIframe = document.createElement('iframe');
            sidebarIframe.src = 'http://localhost:8502/';
            sidebarIframe.style.position = 'fixed';
            sidebarIframe.style.top = '0';
            sidebarIframe.style.right = '0';
            sidebarIframe.style.width = '300px';
            sidebarIframe.style.height = '100%';
            sidebarIframe.style.border = '1px solid #ccc';
            sidebarIframe.style.zIndex = '9999';
            document.body.appendChild(sidebarIframe);
        }
    }
});