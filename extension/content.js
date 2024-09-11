function getPageContent() {
    return {
      title: document.title,
      url: window.location.href,
      content: document.body.innerText
    };
  }
  
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getTabContent') {
      sendResponse(getPageContent());
    }
    return true;  // This line is important for asynchronous response
  });
  
  // Immediately send a message when the content script loads
  chrome.runtime.sendMessage({action: 'contentScriptLoaded'});