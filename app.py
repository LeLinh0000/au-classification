"""
    Created by nguyenvanhieu.vn at 9/16/2018
"""
from flask import Flask, render_template, redirect, url_for, request
from flask_restful import Resource, Api
from predict_img import recognize_organs_fc_img
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
api = Api(app)

app.config['UPLOAD_FOLDER'] = './static/img/input'

class Predict(Resource):
    def get(self):
        result = recognize_organs_fc_img()
        return result

api.add_resource(Predict, '/api/predict')

@app.route('/')
def welcome():
    return render_template('classify.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['coverImage']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
      return 'file uploaded successfully'

# @app.route("/api/predict")
# def get_label():
#     result = recognize_organs_fc_img()
#     return result

# @app.route('/home')
# def home():
#     return 'Login success!'


# # Route for handling the login page logic
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         if request.form['username'] != 'admin' or request.form['password'] != 'admin':
#             error = 'Invalid Credentials. Please try again.'
#         else:
#             return redirect(url_for('home'))
#     return render_template('login.html', error=error)


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
