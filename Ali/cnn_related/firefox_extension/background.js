
// setInterval(function(){
//     console.log("in background script")
// }, 1000);

//set up the port connection: 

console.log("starting backround script")
//get ready to get the image and then take a screen shot, 
//then post it to the front end content script
var imageName = undefined; 
var portFromCS;
function connected(p) {
  portFromCS = p;
  portFromCS.postMessage({greeting: "hi there content script!", type: "greeting"});
  
  portFromCS.onMessage.addListener(function(m) {
   if(m.type =="greeting"){
        console.log("In background script, received message from content script")
        console.log(m.greeting);
   }else if (m.type =="screenshot"){
       imageName = m.name;
       //take screenshot 
       var capturing = browser.tabs.captureVisibleTab();
       capturing.then(onCaptured, onError);
       
   }

  });
}
browser.runtime.onConnect.addListener(connected);



function onCaptured(imageUri) {
//   console.log("have capture1");
//   console.log(imageUri);
//   console.log("have captured2")
     portFromCS.postMessage({name: imageName, image:imageUri, type: "screenshot"});
}

function onError(error) {
//   console.log("error")
  console.log(`Error: ${error}`);
}

// var capturing = browser.tabs.captureVisibleTab();
// capturing.then(onCaptured, onError);
// console.log(capturing);  



