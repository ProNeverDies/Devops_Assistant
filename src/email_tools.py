from imap_tools import MailBox, AND
from langchain_core.tools import tool
import re

def get_imap_config(assistant):
    return (
        assistant.config.get("imap_host"),
        assistant.config.get("imap_user"),
        assistant.config.get("imap_password")
    )

@tool
def list_unread_emails(limit: int = 5) -> str:
    """Fetch unread emails via IMAP. Returns UID, sender, subject, and date."""
    from devops import assistant
    raw_llm = assistant.raw_llm
    host, user, pw = get_imap_config(assistant)

    if not all([host, user, pw]):
        return " Email config missing."

    try:
        with MailBox(host).login(user, pw, initial_folder='INBOX') as mb:
            unread = list(mb.fetch(criteria=AND(seen=False), headers_only=True, limit=limit))
        if not unread:
            return " No unread emails."

        result = []
        for mail in unread:
            result.append(
                f"UID: {mail.uid}\\nFrom: {mail.from_}\\nSubject: {mail.subject}\\nDate: {mail.date}\\n"
            )
        return '\\n'.join(result)
    except Exception as e:
        return f" Error: {str(e)}"

@tool
def summarize_email(uid: str) -> str:
    """Summarize one email given its UID using LLM."""
    from devops import assistant
    raw_llm = assistant.raw_llm 
    host, user, pw = get_imap_config(assistant)

    if not all([host, user, pw]):
        return " Email config missing."
    if not uid or not uid.isdigit():
        return "Invalid UID."

    try:
        with MailBox(host).login(user, pw, initial_folder='INBOX') as mb:
            mail = next(mb.fetch(AND(uid=uid), mark_seen=False), None)
        if not mail:
            return f" No email with UID {uid}."

        body = mail.text or (re.sub('<[^<]+?>', '', mail.html or "") if mail.html else "No content")
        prompt = (
            f"Summarize this email:\n\nSubject: {mail.subject}\nFrom: {mail.from_}\nDate: {mail.date}\n\n{body[:500]}"
        )

        return raw_llm.invoke(prompt).content
    except Exception as e:
        return f" Failed to summarize UID {uid}: {e}"

if __name__ == "__main__":
    from devops import assistant
    raw_llm = assistant.raw_llm 
    print(" Testing list_unread_emails:")
    print(list_unread_emails({"limit":1}))

    print("\n Testing summarize_email (example UID: 123):")
    print(summarize_email("123"))


