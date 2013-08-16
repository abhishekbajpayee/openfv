import smtplib
from email.mime.text import MIMEText

def send_email(to, subject, body):

    msg = MIMEText(body)
    msg['From'] = 'ehlatmit@gmail.com'
    msg['To'] = to
    msg['Subject'] = subject

    server = smtplib.SMTP('smtp.gmail.com',587) #port 465 or 587
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login('ehlatmit@gmail.com','ehlwave2013')
    server.sendmail('ehlatmit@gmail.com', to, msg.as_string())
    server.close()
