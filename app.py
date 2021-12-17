import os
from math import floor
from flask import Flask, render_template, request
from models import HAM10000_Densenet

app = Flask(__name__)

image_directory = "temp_images"

model = HAM10000_Densenet()

@app.route('/')
def index():
    return render_template('upload-form.html')

@app.route('/upload', methods=['POST'])
def success():
    if request.method == 'POST':
        # save the file
        f = request.files['file']
        localPath = os.path.join(image_directory, f.filename)
        f.save(localPath)
        # make prediction
        confidence = model.evaluate(localPath, app.logger)
        # delete file
        os.remove(localPath)
        # get the confidence as a percentage
        confidence = floor(confidence * 10000) / 100
        # show the result
        return render_template('result.html', confidence=confidence)


if __name__ == '__main__':
    ip = '0.0.0.0'
    port = int(os.environ.get("PORT", 80))
    app.run(host=ip, port=port)
