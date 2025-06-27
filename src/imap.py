import imaplib

EMAIL = "xyz@gmail.com"
PASSWORD = "################"  

try:
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")
    print(" IMAP is working!")
except Exception as e:
    print(" IMAP failed:", e)
