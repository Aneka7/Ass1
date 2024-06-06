from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data
        pwb = float(request.form['pwb'])
        ewb = float(request.form['ewb'])
        rwb = float(request.form['rwb'])
        nse = float(request.form['nse'])
        pss = float(request.form['pss'])
        soc = float(request.form['soc'])
        
        # Calculate mental well-being
        mwb = ((pwb + ewb + rwb) - (nse + pss)) / 3 + soc
        
        return render_template('calc.html', mwb=mwb)
    
    return render_template('calc.html', mwb=None)

if __name__ == '__main__':
    app.run(debug=True)
