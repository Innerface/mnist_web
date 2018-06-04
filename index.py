from flask import Flask
from flask import render_template
import os
import sys
sys.path.append("../tensorflow")
import deep_mnist
import json
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

app = Flask(__name__)  
     
@app.route("/")  
def hello():  
    return "Hello World!"  
      
@app.route("/mnist")
def mnist():
    result = deep_mnist.predict()
    return json.dumps(result.tolist())
    
@app.route("/word/<word>")
def word(word):
    model = gensim.models.Word2Vec.load("./static/data/wiki.zh.text.ltp.model")
    result = model.most_similar(positive=[word])
    # return json.dumps(result)
    return render_template('word.html',result=result)

@app.route("/image/")
@app.route("/image/<num>")
def image(num):
    url = "../static/image/train_"+num+".bmp"
    num = int(num)
    result = deep_mnist.predict(num)
    result = json.dumps(result.tolist())
    return render_template('image.html',num=num,url=url,result=result)

if __name__ == "__main__":  
    app.run()  