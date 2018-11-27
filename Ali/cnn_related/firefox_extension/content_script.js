/*
Just draw a border round the document.body.
*/
document.body.style.border = "5px solid red";




//Main set up to connect to the long running background script
//after getting a screenshot -> we will then request the next data 
var myPort = browser.runtime.connect({name:"port-from-cs"});
myPort.postMessage({greeting: "hello from content script", type:"greeting"});

myPort.onMessage.addListener(function(m) {
    console.log(m);
   if(m.type =="greeting"){
         console.log("In content script, received message from background script: ");
         console.log(m.greeting);
   }else if (m.type =="screenshot"){
       var imageName  = m.name; 
       var imageData = m.image; 
       console.log("received screenshot...")
       console.log(imageData);
       //save to the backend 
       fetch('http://127.0.0.1:5000/storeImage', { 
        method: 'POST', 
        headers: {'Content-Type': 'application/json; charset=utf-8'}, 
        body: JSON.stringify({"name": imageName, "image": imageData})
        })
        .then(res => {return res.json()})
        .then(data => {if(data.success !="true"){ alert("not working!") }} )
       
   }

});





function getScreenShot(nameOfImage){
    myPort.postMessage({name: nameOfImage, type: "screenshot"});
}

setTimeout(function(){
    if(!window.wrappedJSObject.map_){
        console.log("window.map_ not found")
        return ;
    }
    if(!window.wrappedJSObject.addPointToMap){
        console.log("window.addPointToMap not found")
        return ;
    }
    
    fetch('http://127.0.0.1:5000/dynamic_data?year=2017&week=31&categoryCode=32', { 
    method: 'GET', 
    headers: {'Content-Type': 'application/json; charset=utf-8'}, 
    })
    .then(res => {return res.json()}).then(data => {
        console.log(data[0]);
        for(var ii = 0;ii<data.length; ii++){
            var long_ = data[ii]['Longitude']
            var lat_ = data[ii]['Latitude']
            // var radius_ = data[ii]['Longitude']
            // console.log("here");
           window.wrappedJSObject.addPointToMap(long_, lat_, '#FF0000', 70 );
            // console.log("here2");
        }        
    });

    console.log("sending screenshot...")
    getScreenShot("nameOfImage")
    
}, 3000);



// https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabs/captureVisibleTab




////// Code that is no longer needed 
// var citymap = {
//             chicago: {
//                 center: {lat: 41.878, lng: -87.629},
//                 population: 2714856
//             },
//             newyork: {
//                 center: {lat: 40.714, lng: -74.005},
//                 population: 8405837
//             },
//             losangeles: {
//                 center: {lat: 34.052, lng: -118.243},
//                 population: 3857799
//             },
//             vancouver: {
//                 center: {lat: 49.25, lng: -123.1},
//                 population: 603502
//             }
//             };
//   for (var city in citymap) {
//     window.wrappedJSObject.addPointToMap(citymap[city].center.lng, citymap[city].center.lat,'#FF0000', 70)
//   }