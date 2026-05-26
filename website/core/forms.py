from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, BooleanField, SelectMultipleField
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
    regions = SelectMultipleField(
        "Target regions",
        choices=[
            ("us", "United States"),
            ("canada", "Canada"),
            ("australia", "Australia"),
            ("new_zealand", "New Zealand"),
            ("uk", "United Kingdom"),
            ("europe_english", "Non-English speaking Europe, English-only roles"),
        ],
        default=["us"],
    )
    require_h1b = BooleanField("Require H1B Sponsorship?")
    always_prompt = BooleanField("Always Prompt for New Results?")
    submit = SubmitField("Pitch Me")
