# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 23:03:42 2021

@author: Sheshank
"""

#%%
//for communication with server
<script>
function loadDoc() {
  const xhttp = new XMLHttpRequest();
  xhttp.onload = function() {
    document.getElementById("demo").innerHTML =
    this.responseText;
  }
  xhttp.open("GET", "user_input.txt");
  xhttp.send();
}
</script>




#%%
//for autoscroll option
var elmnt = document.getElementById("content");

function scrollToTop() {
  elmnt.scrollIntoView(true); // Top
}

function scrollToBottom() {
  elmnt.scrollIntoView(false); // Bottom
} 

#%%
// Add element to the chat window as a pargraph
var para = document.createElement("P");                       // Create a <p> node
var t = document.createTextNode("This is a paragraph.");      // Create a text node
para.appendChild(t);                                          // Append the text to <p>
document.getElementById("myDIV").appendChild(para);

