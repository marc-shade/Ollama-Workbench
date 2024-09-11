document.addEventListener('DOMContentLoaded', () => {
    const ollamaChatIframe = document.getElementById('ollamaChat');
    
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const currentTab = tabs[0];
      if (currentTab) {
        let iframeUrl = new URL('http://localhost:8502/');
        iframeUrl.searchParams.append('url', currentTab.url);
        iframeUrl.searchParams.append('extension', 'true');
        ollamaChatIframe.src = iframeUrl.toString();
      } else {
        console.error("No active tab found");
        ollamaChatIframe.src = 'http://localhost:8502/?extension=true';
      }
    });
  });