
//console.log("First executed Script");
document.addEventListener("DOMContentLoaded", css_set);
//
var chat_server_entry;
//
var chat_server_div;
//
var chat_user_entry;
//
var chat_user_div;
//
var user_put="";
var server_put="";
var awaiting_response=false; // For debugging
var animate_element;
//
// Elemental containers
var maker_element = document.getElementById("makers");
var chat_element = document.getElementById("chat");
var parent_element = document.getElementById("parent");
//windows within chat elemental container
var chat_window = document.getElementById("chat_record");
var text_window = document.getElementById("chat");

//console.log("The Window width is : ",window.innerWidth);
//console.log("The Window Height is : ",window.innerHeight);
//document.onload=css_set;
//console.log(maker_element.style);
create_animate_element();
window.addEventListener("resize", window_resize);
function window_resize()
{
    css_set();
}
function create_animate_element()
{
    //console.log("Animate element is created")
    animate_element = document.createElement("p");
    animate_element.setAttribute("id", "animate");
    var dots = document.createElement("span");
    dots.setAttribute("id", "dots");
    animate_element.appendChild(document.createTextNode("Waiting for server"));
    animate_element.appendChild(dots);
}
function clear_variables()
{
    //user entry
    chat_user_entry = document.createElement("P");
    chat_user_entry.className="chat_user_entry";
    chat_user_div = document.createElement("div");
    chat_user_div.className="chat_user_div";
    //server entry 
    chat_server_entry = document.createElement("P");
    chat_server_entry.className="chat_server_entry";
    chat_server_div = document.createElement("div");
    chat_server_div.className="chat_server_div";

}
function enterValue(val, cls)
 {
    var entry;
    var divs;
    //console.log("Entered enterValue Function");
    //console.log(val,cls);
    if (cls == "uin") 
    {
        //Change here so that Text node is changed to paragraph node
        //console.log("User input given (User) :", val);
        var text_node=document.createElement("p");
        //text_node.style.className = "chat_user_entry";
        //text_node.style.align="right";
        text_node.appendChild(document.createTextNode(val + "\n"));
        entry=chat_user_entry.appendChild(text_node);
        //console.log("User input given :", entry);
        divs = chat_user_div;
        divs.appendChild(entry);
        
    }
    else if (cls=="sin") 
    {
        // Change here so that Text node is changed to paragraph node
        //console.log("User input given (Server) :", val);
        var text_node = document.createElement("p");
        //text_node.style.align="left";
        //text_node.style.className = "chat_server_entry";
        text_node.appendChild(document.createTextNode(val + "\n"));
        entry = chat_server_entry.appendChild(text_node);
        //console.log("User input given :", entry);
        divs=chat_server_div;
        divs.appendChild(entry);

    }
    else if (cls=="animate")
    {
        entry = chat_server_entry.appendChild(animate_element);
        divs=chat_server_div;
        divs.appendChild(entry);
        divs.setAttribute("id", "anim");
    }
    document.getElementById("chat_record").appendChild(divs);
    clear_variables();
}
function submitValue()
{
    //console.log("Entered submitValue Function");
    //console.log(document.getElementById("user_input").value);
    //console.log("user_put Variable before : " + user_put);
    user_put = document.getElementById("user_input").value;
    //var form=$("#")
    // Store this variable in local storage for future analysis
    //console.log("user_put Variable after : " + user_put);
    //Following state is entirely for debugging purposes. Remove it when not needed.
    enterValue(user_put, "uin");
    console.log("ABOUT to send to server :",user_put);
    //var k = $("#user_input").serialize();
    //console.log("What the server is expecting",k)
    //user_put = XML_out (user_put);
    //user_put="uin="+user_put;
    send_to_server(user_put);
}
function XML_out(text_input)
{
    var xmlDoc = document.implementation.createDocument(null,"user_input");
    console.log(xmlDoc);
    text_input = "{"+
    "type:"+ "POST,"+
    "url :"+ "/uin,"+
    "data :"+
    text_input+
    "}";
    console.log("Text input after processing" ,text_input);
    return text_input;
}
//should add a document ready function that updates the element sizes.
function css_set() 
{
    //console.log("css setting happening")
    parent_element.style.display="flex";
    parent_element.style.alignItems="stretch";
    //document.getElementById("makers").style.display="inline-flex";
    //document.getElementById("chat").style.display = "inline-flex";
    var css_width=window.innerWidth;
    var css_height=window.innerHeight;
    //console.log(css_width);
    //console.log(css_height);
    parent_element.style.width=String(css_width-20)+"px";
    parent_element.style.height=String(css_height*8/10-40)+"px";
    //maker_element setting
    maker_element.style.width = "25%";
    maker_element.style.height = "90%";
    maker_element.style.margin="10px";
    maker_element.style.paddingTop=String(parent_element.style.width/4)+"px";
    maker_element.style.paddingBottom=String(parent_element.style.height)+"px";
    maker_element.style.paddingRight="10px";
    maker_element.style.paddingLeft="10px";
    maker_element.style.borderStyle="solid";
    maker_element.style.borderColor="#92a8d1";
    //chat_element setting
    chat_element.style.width = "75%";
    chat_element.style.height = "90%";
    chat_element.style.margin = "10px";
    chat_element.style.paddingTop = String(parent_element.style.width / 4) + "px";
    chat_element.style.paddingBottom = String(parent_element.style.height) + "px";
    chat_element.style.paddingRight = "10px";
    chat_element.style.paddingLeft = "10px";
    chat_element.style.borderStyle = "solid";
    chat_element.style.borderColor = "#92a8f5";
    //maker_element.style.boxSizing="100px";
    clear_variables();
}

function send_to_server(uin) 
{
    // Here write the code such that it ignores the request until current request is processed.
    var xhttp = new XMLHttpRequest();
    xhttp.open("POST","/uin",true);
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    if (awaiting_response==true)
    {
        // Write code here for further checks
        console.log("Awaiting server response, so info will not be sent");
    }
    else 
    {
        awaiting_response=true;
        xhttp.send("data="+uin);
        enterValue(server_put,"animate");
        console.log("ABOUT to send outside the function", uin);
        xhttp.onload=function()
        {
        server_put = xhttp.responseText;
        console.log("Server Response Received :",server_put);
        var ele = document.getElementById("animate");
        awaiting_response = false;
        //console.log("This is the element", ele);
        ele.remove();
        var ele = document.getElementById("anim");
        //console.log("This is the Div that needs eliminated", ele)
        ele.remove();
        //document.getElementsByClassName("chat_entry").removeChild(ele);
        console.log("Function is reaching here :",server_put);
        enterValue(server_put, "sin");
        }
    }
}