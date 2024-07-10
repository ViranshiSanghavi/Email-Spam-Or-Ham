import streamlit as st
import pickle
import string
import imaplib
import email
from email.header import decode_header
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd

ps = PorterStemmer()


nltk.download('stopwords')
nltk.download('punkt')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def fetch_emails(email_user, email_pass, num_emails=50):
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(email_user, email_pass)
    mail.select("inbox")

    result, data = mail.search(None, "ALL")
    mail_ids = data[0]
    id_list = mail_ids.split()

    emails = []
    fetched_count = 0

    for num in id_list:
        if fetched_count >= num_emails:
            break

        try:
            result, data = mail.fetch(num, "(RFC822)")
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject = decode_header(msg["Subject"])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()

            from_ = msg.get("From")

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    try:
                        body = part.get_payload(decode=True).decode()
                    except:
                        body = None
                    if content_type == "text/plain" and "attachment" not in content_disposition and body:
                        emails.append((subject, from_, body))
                        fetched_count += 1
                        break
            else:
                content_type = msg.get_content_type()
                body = msg.get_payload(decode=True).decode()
                if content_type == "text/plain" and body:
                    emails.append((subject, from_, body))
                    fetched_count += 1

        except Exception as e:
            st.error(f"Error fetching email {num}: {e}")

    mail.logout()
    st.write(f"Fetched: {fetched_count} emails")
    return emails[:num_emails]

st.title("Email Spam Classifier")


with st.expander("Fetch and Classify Emails"):
    email_user = st.text_input("Enter your email")
    email_pass = st.text_input("Enter your app password", type="password")
    num_emails = st.number_input("Number of emails to fetch", min_value=1, max_value=500, value=50)

    if st.button('Fetch Emails'):
        try:
            emails = fetch_emails(email_user, email_pass, num_emails)
            email_data = []

            for subject, from_, body in emails:
                transformed_sms = transform_text(body)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                spam_status = "Spam" if result == 1 else "Not Spam"
                email_data.append({"From": from_, "Subject": subject, "Spam Status": spam_status})

            df = pd.DataFrame(email_data)
            st.dataframe(df)
        except imaplib.IMAP4.error as e:
            st.error(f"Authentication failed: {e}")


st.write("---")
st.header("Manual Spam Check")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")