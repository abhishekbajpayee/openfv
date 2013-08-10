import smtplib

def send_email(to, subject, body):

    msg = "\r\n".join([
            "From: ehlatmit@gmail.com",
            "To: ", to,
            "Subject: ", subject,
            "",
            body
            ])

    server = smtplib.SMTP('smtp.gmail.com',587) #port 465 or 587
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login('ehlatmit@gmail.com','ehlwave2013')
    server.sendmail('ehlatmit@gmail.com', to, body)
    server.close()
