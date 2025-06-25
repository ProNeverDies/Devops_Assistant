from imap_tools import MailBox

EMAIL_USER = 'devansh.tomar2022@vitstudent.ac.in'
EMAIL_PASSWORD = 'cdmhdrnrjwdfwang'

with MailBox('imap.gmail.com').login(EMAIL_USER, EMAIL_PASSWORD) as mailbox:
    for msg in mailbox.fetch(limit=1):
        print(f"From: {msg.from_}, Subject: {msg.subject}")
