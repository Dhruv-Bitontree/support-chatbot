(function () {
  "use strict";

  var config = {
    apiUrl:
      document.currentScript?.getAttribute("data-api-url") ||
      window.location.origin,
  };

  // Create widget container
  var container = document.createElement("div");
  container.id = "support-chatbot-widget";
  container.style.cssText =
    "position:fixed;bottom:24px;right:24px;z-index:99999;font-family:system-ui,sans-serif;";

  // Chat iframe (hidden initially)
  var iframe = document.createElement("iframe");
  iframe.src = config.apiUrl + "/chat?embed=true";
  iframe.style.cssText =
    "width:380px;height:560px;border:none;border-radius:16px;box-shadow:0 25px 50px -12px rgba(0,0,0,0.25);" +
    "display:none;margin-bottom:12px;background:white;";
  iframe.setAttribute("title", "Customer Support Chat");

  // Toggle button
  var button = document.createElement("button");
  button.style.cssText =
    "width:56px;height:56px;border-radius:50%;border:none;cursor:pointer;display:flex;" +
    "align-items:center;justify-content:center;background:#2563eb;box-shadow:0 10px 15px -3px rgba(37,99,235,0.3);" +
    "transition:background 0.2s;float:right;";
  button.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="24" height="24">' +
    '<path fill-rule="evenodd" d="M4.848 2.771A49.144 49.144 0 0112 2.25c2.43 0 4.817.178 7.152.52 1.978.292 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.678-3.348 3.97a48.901 48.901 0 01-3.476.383.39.39 0 00-.297.17l-2.755 4.133a.75.75 0 01-1.248 0l-2.755-4.133a.39.39 0 00-.297-.17 48.9 48.9 0 01-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97z" clip-rule="evenodd"/>' +
    "</svg>";

  var isOpen = false;
  button.addEventListener("click", function () {
    isOpen = !isOpen;
    iframe.style.display = isOpen ? "block" : "none";
    button.style.background = isOpen ? "#4b5563" : "#2563eb";
    button.innerHTML = isOpen
      ? '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="24" height="24"><path fill-rule="evenodd" d="M5.47 5.47a.75.75 0 011.06 0L12 10.94l5.47-5.47a.75.75 0 111.06 1.06L13.06 12l5.47 5.47a.75.75 0 11-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 01-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 010-1.06z" clip-rule="evenodd"/></svg>'
      : '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="24" height="24"><path fill-rule="evenodd" d="M4.848 2.771A49.144 49.144 0 0112 2.25c2.43 0 4.817.178 7.152.52 1.978.292 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.678-3.348 3.97a48.901 48.901 0 01-3.476.383.39.39 0 00-.297.17l-2.755 4.133a.75.75 0 01-1.248 0l-2.755-4.133a.39.39 0 00-.297-.17 48.9 48.9 0 01-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97z" clip-rule="evenodd"/></svg>';
  });

  button.addEventListener("mouseenter", function () {
    button.style.background = isOpen ? "#374151" : "#1d4ed8";
  });
  button.addEventListener("mouseleave", function () {
    button.style.background = isOpen ? "#4b5563" : "#2563eb";
  });

  container.appendChild(iframe);
  container.appendChild(button);
  document.body.appendChild(container);
})();
