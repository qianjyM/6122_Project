from flask import Flask, request, render_template
import socket
import time
#This is a test
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == 'wav'

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 处理上传的音频文件
        if 'audio' not in request.files:
            return 'No file part', 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return 'No selected file', 400

        print(audio_file)
        # 调用模型处理音频，并计算处理时间
        if audio_file and allowed_file(audio_file.filename):
            start_time = time.time()
            result = process_audio(audio_file)
            end_time = time.time()
            processing_time = end_time - start_time
            return render_template('result.html', result=result, processing_time=processing_time)
        else:
            return render_template('upload.html', error='Only .wav files are allowed!')
    return render_template('upload.html')

def process_audio(file):
    # 这里放置处理音频的代码
    # 模拟返回结果
    return "Yes" # 或 "No"

if __name__ == '__main__':
    app.run(debug=True,port=9007)
