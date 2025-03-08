// Function to check server availability
async function checkServer(port) {
    try {
        console.log(`Checking Streamlit server on port ${port}...`);
        
        // Create an iframe to test the connection
        const testIframe = document.createElement('iframe');
        testIframe.style.display = 'none';
        document.body.appendChild(testIframe);
        
        return new Promise((resolve, reject) => {
            testIframe.onload = () => {
                document.body.removeChild(testIframe);
                console.log('Streamlit server is accessible');
                resolve(true);
            };
            
            testIframe.onerror = () => {
                document.body.removeChild(testIframe);
                console.log('Failed to load Streamlit server');
                resolve(false);
            };
            
            testIframe.src = `http://127.0.0.1:${port}/?extension=true`;
        });
    } catch (error) {
        console.log(`Server check failed on port ${port}:`, error.message);
        return false;
    }
}

// Function to update connection status
async function updateStatus(port) {
    const statusDiv = document.getElementById('status');
    try {
        console.log(`Checking Streamlit connection on port ${port}`);
        const isConnected = await checkServer(port);
        if (isConnected) {
            statusDiv.textContent = '🟢 Connected';
            statusDiv.style.color = '#4CAF50';
            return true;
        }
    } catch (error) {
        console.error('Connection check failed:', error.message);
    }
    statusDiv.textContent = '🔴 Not Connected';
    statusDiv.style.color = '#f44336';
    return false;
}

// Function to load the iframe
function loadIframe(port) {
    const ollamaChatIframe = document.getElementById('ollamaChat');
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const currentTab = tabs[0];
        if (currentTab) {
            const urlParams = new URLSearchParams(window.location.search);
            const selectedText = urlParams.get('text');
            
            let iframeUrl = new URL(`http://127.0.0.1:${port}/`);
            if (selectedText) {
                iframeUrl.searchParams.append('selected_text', selectedText);
            }
            if (currentTab.url) {
                iframeUrl.searchParams.append('url', currentTab.url);
            }
            iframeUrl.searchParams.append('extension', 'true');
            
            console.log(`Loading iframe with URL: ${iframeUrl.toString()}`);
            ollamaChatIframe.src = iframeUrl.toString();
            
            // Add load event listener to iframe
            ollamaChatIframe.onload = () => {
                console.log('Iframe loaded successfully');
                ollamaChatIframe.style.display = 'block';
                document.getElementById('error').style.display = 'none';
            };
            
            ollamaChatIframe.onerror = (error) => {
                console.error('Iframe failed to load:', error);
                document.getElementById('error').style.display = 'block';
                ollamaChatIframe.style.display = 'none';
            };
        } else {
            console.error("No active tab found");
            ollamaChatIframe.src = `http://127.0.0.1:${port}/?extension=true`;
        }
    });
}

// Initial connection attempt with retries
async function connectWithRetry(maxRetries = 5) {
    const statusDiv = document.getElementById('status');
    const ollamaChatIframe = document.getElementById('ollamaChat');
    const errorDiv = document.getElementById('error');
    
    statusDiv.textContent = '🟡 Connecting...';
    statusDiv.style.color = '#FFA500';
    
    console.log('Attempting to connect to Streamlit server...');
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        console.log(`\nConnection attempt ${attempt + 1} of ${maxRetries}`);
        
        const isConnected = await checkServer(8501);
        if (isConnected) {
            console.log('Successfully connected to Streamlit server');
            loadIframe(8501);
            
            // Set up periodic status check
            setInterval(() => updateStatus(8501), 5000);
            
            // Hide error div if it was shown
            errorDiv.style.display = 'none';
            return;
        }
        
        console.log(`Attempt ${attempt + 1} failed, waiting before retry...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    // If all retries failed
    console.log('All connection attempts failed');
    console.log('Please check:');
    console.log('1. Is Ollama Workbench running? (http://127.0.0.1:8501)');
    console.log('2. Try accessing http://127.0.0.1:8501 in your browser');
    console.log('3. Check for any CORS or network errors in the console');
    
    statusDiv.textContent = '❌ Connection Failed';
    statusDiv.style.color = '#f44336';
    ollamaChatIframe.style.display = 'none';
    errorDiv.style.display = 'block';
}

// Start connection process when the document is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Popup loaded, starting connection process...');
    
    // Add retry button handler
    document.getElementById('retryButton').addEventListener('click', () => {
        console.log('Retry button clicked, attempting reconnection...');
        document.getElementById('error').style.display = 'none';
        document.getElementById('ollamaChat').style.display = 'block';
        connectWithRetry();
    });
    
    // Start initial connection
    connectWithRetry();
});