// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

'use strict';

// chrome.runtime.onInstalled.addListener(function() {
  // chrome.storage.sync.set({color: '#3aa757'}, function() {
    // console.log("The color is green.");
  // });
// });


//control shift k  and control k 

// chrome.runtime.onMessage.addListener(
  // function(request, sender, sendResponse) {
    // console.log(sender.tab ?
                // "from a content script:" + sender.tab.url :
                // "from the extension");
    // if (request.greeting == "hello")
      // sendResponse({farewell: "goodbye"});
  // });
  
    // chrome.runtime.onMessage.addListener(
    // function(message, callback) {
      // if (message == “changeColor”){
        // chrome.tabs.executeScript({
          // code: 'document.body.style.backgroundColor="orange"'
        // });
      // }
   // });
   
   
 chrome.runtime.onConnect.addListener(function(port) {
  console.assert(port.name == "update_map");
  console.log("starting ")
  port.postMessage({ type: "FROM_CONTENT_SCRIPT", text: "Hello from the content script!" });
  

  // port.onMessage.addListener(function(msg) {
	//   // console.log(msg);
	//   port.postMessage({ type: "FROM_CONTENT_SCRIPT", text: "Hello from the content script!" });
  // });
});
























// chrome.runtime.onConnect.addListener(function(port) {
//   console.assert(port.name == "knockknock");
//   console.log("starting ")
//   port.onMessage.addListener(function(msg) {
// 	  // console.log(msg);
	  
//     chrome.tabs.captureVisibleTab(null,{},function(dataUri){
//       // console.log(dataUri);
//       port.postMessage({question: dataUri});
//   });
	  
	  
//     // if (msg.joke == "Knock knock")
//     //   port.postMessage({question: "Who's there?"});
//     // else if (msg.answer == "Madame")
//     //   port.postMessage({question: "Madame who?"});
//     // else if (msg.answer == "Madame... Bovary")
//     //   port.postMessage({question: "I don't get it."});
//   });
// });