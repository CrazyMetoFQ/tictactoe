from tictactoe import Board
from flask import Flask, render_template, send_file, request, json


app = Flask(__name__)
app.config['SECRET_KEY'] = 'damn this i shard'


# @app.route('/favicon.ico')
# def favicon():
#     return send_file(f"{os.getcwd()}/static/favicon.ico")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play/<n>')
def play(n):
    
    n = int(n)
    w=None
    
    if n<=3: w=n
    else: w=4
  

    global brd
    brd = Board(n, w)
    return render_template('tictactoe.html', n=n)

@app.route('/submit', methods=['GET','POST'])
def submit():
    data = request.get_json()
    
    brd.change(brd=data)
    wonX = brd.wincheck("X")
    wonO = brd.wincheck("O")
    if wonX: w_resp = [True,"X"]
    elif wonO: w_resp = [True,"O"]
    else:
        pass
    
    if wonX or wonO:
        response = app.response_class(
                    response=json.dumps(w_resp),
                    status=200,
                    mimetype='application/json'
                    )
                
    else:
        response = app.response_class(
                    response=json.dumps(data),
                    status=200,
                    mimetype='application/json'
                    )
        
    return response

if __name__ == '__main__':
  app.run(port=5000, debug=True
          # ,host='0.0.0.0'
 )
 