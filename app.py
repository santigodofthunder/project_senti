from flask import Flask,render_template,redirect,url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from sentiment import vectr,LR_from_joblib

app=Flask(__name__)

app.config['SECRET_KEY']='mysecretkey'

class SentimentForm(FlaskForm):
      review=StringField("How is the food?")
      submit = SubmitField('Submit')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/sentiment',methods=['GET','POST'])
def page_sentiment():
    review=False
    form=SentimentForm()
    if form.validate_on_submit():
        predicted_code=LR_from_joblib.predict(vectr.fit_transform([form.review.data]))
        if predicted_code[0]==1:
            review='Positvie'
        else:
            review='Negative'
        form.review.data=''
    return render_template('sentiment.html',review=review,form=form)

if __name__ == '__main__':
    app.run()
