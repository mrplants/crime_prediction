// alert("hello content script");

var port = chrome.runtime.connect({name: "knockknock"});


setTimeout(function(){
  console.log("starting");
  port.postMessage({joke: "Knock knock"});
}, 3000);


port.onMessage.addListener(function(msg) {
  console.log(msg.question);

  fetch('https://jsonplaceholder.typicode.com/todos/1')
  .then(response => response.json())
  .then(json => console.log(json))

  
  // window.run_123();
  // if (msg.question == "Who's there?")
    // port.postMessage({answer: "Madame"});
  // else if (msg.question == "Madame who?")
    // port.postMessage({answer: "Madame... Bovary"});
});



// chrome.tabs.captureVisibleTab(null, {}, function (image) {
//   // You can add that image HTML5 canvas, or Element.
//   var img = document.getElementById('imgID');
//   img.src = image;
//   console.log(crome.tabs);
  
// });