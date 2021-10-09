import os
from flask import Flask, render_template, url_for, request, jsonify, session
import time

print(os.getcwd())
app = Flask(__name__)
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
def user_requests(uin):
    """User input is handled as an XML file from the POST mechanism"""
    print("request recieved")
    if request.method == "POST":
        print("identified as POST request")
        try:
            print(request.args)
        except:
            pass
        print("received packets as it is")
        print(uin)
    return uin
    
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


######################################################################
# Running the Basic Flask app here
######################################################################
if __name__=='__main__':
    app.run()


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
