import smtplib
from email.message import EmailMessage

class EmailMessenger:
    def __init__(self, user, password, email):
        self.user = user
        self.password = password
        self.email = email

    def send_email(self, subject, body, to):
        msg = EmailMessage()
        msg.set_content(body)
        msg['subject'] = subject
        msg['to'] = to
        msg['from'] = self.email
        server = None
        try:
            server = smtplib.SMTP("email.infineon.com", 587, timeout=10)
            server.starttls()
            print("E-Mail Server Connection Successful.")
            server.login(self.user, self.password)
            server.send_message(msg)
        except Exception as e:
            print("E-Mail Connection Failed.", e)
        finally:
            if server:
                server.quit()
