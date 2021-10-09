



#%%
import os
from flask import Flask, render_template, url_for, request, jsonify, session
import time
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import dialogue
import pandas as pd
#%%
print(os.getcwd())
app = Flask(__name__)
global chat
chat=dialogue.chat_interface_handler()
#%%
################################################################
# redirect - used to call the decorated function with appropriate parameters passed.
######################################################################
# Function rendering simple template
######################################################################
@app.route('/')
def home() :
    return render_template("home.html")

######################################################################
# End of Function
######################################################################
######################################################################
# Function accepting simple post and examining it to produce output
######################################################################
@app.route("/uin",methods=["POST"])
def user_requests():
    """User input is handled as an XML file from the POST mechanism"""
    print("Request : recieved")
    if request.method == "POST":
        print("Identified as request : POST")
        #print(type(request))
        #print(str(request))
        print("Request is :",request)
        try:
            #print("Request Arguments :",request.args)
            #print("Request form Keys :",request.form.keys())
            inp=request.form["data"]
            txt=chat.dialog_in(inp)
            return txt
        except:
            print("This has failed")
            return "There is something wrong with Data"
        #print("received packets as it is")
        #print(uinput.xml")
    
    
######################################################################
# End of Function
######################################################################

######################################################################
# Function accepting simple close request
######################################################################
@app.route("/close",methods=["POST"])
def close():
    """"Closes all the functions """
    pass
######################################################################
#
######################################################################

######################################################################
#
######################################################################
#
######################################################################
#%%

######################################################################
# Running the Basic Flask app here
######################################################################
if __name__=='__main__':
    app.run()


#%%
"""
Use this function for succesful training message and rediction to server
@app.route('/login', methods=['GET', 'POST'])
def login():
   error = None

   if request.method == 'POST':
      if request.form['username'] != 'admin' or \
         request.form['password'] != 'admin':
         error = 'Invalid username or password. Please try again!'
      else:
         flash('You were successfully logged in')
         return redirect(url_for('index'))
   return render_template('login.html', error=error)"""
