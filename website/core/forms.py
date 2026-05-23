from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Email


class QuestionForm(FlaskForm):
    question = StringField("Question", validators=[DataRequired()])
    submit = SubmitField("Ask")


class EmailForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    subject = StringField("Subject", validators=[DataRequired()])
    body = TextAreaField("Message", validators=[DataRequired()])
    submit = SubmitField("Send")

class JobURLForm(FlaskForm):
    url = StringField("Job Description URL", validators=[DataRequired()])
    require_h1b = BooleanField("Require H1B Sponsorship?")
    submit = SubmitField("Pitch Me")
