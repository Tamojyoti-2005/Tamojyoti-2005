import os
import sys
import site
import sysconfig
# Remove the current script directory from sys.path so local shadowing packages
# under d:\Python do not override installed dependencies.
script_dir = os.path.abspath(os.path.dirname(__file__))
cleaned = []
for entry in sys.path:
    if not entry:
        entry_path = os.path.abspath('.')
    else:
        entry_path = os.path.abspath(entry)
    if entry_path != script_dir:
        cleaned.append(entry)
sys.path[:] = cleaned

# Ensure interpreter site-packages are searched before any remaining local paths.
site_packages = []
if hasattr(site, "getsitepackages"):
    site_packages.extend(site.getsitepackages())
if hasattr(site, "getusersitepackages"):
    site_packages.append(site.getusersitepackages())
site_packages.append(sysconfig.get_paths().get("purelib", ""))
site_packages.append(sysconfig.get_paths().get("platlib", ""))

seen = set()
for p in site_packages:
    if not p:
        continue
    abs_p = os.path.abspath(p)
    if abs_p in seen:
        continue
    seen.add(abs_p)
    if abs_p in sys.path:
        sys.path.remove(abs_p)
    sys.path.insert(0, abs_p)

import speech_recognition as sr
import pyttsx3
import smtplib
import ssl
from email.mime.text import MIMEText
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langdetect import detect

engine = pyttsx3.init()

summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarizer_model.config.max_length = None

generator_tokenizer = AutoTokenizer.from_pretrained("gpt2")
generator_model = AutoModelForCausalLM.from_pretrained("gpt2")
generator_tokenizer.pad_token = generator_tokenizer.eos_token
generator_model.config.max_length = None

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 10.0
    recognizer.non_speaking_duration = 0.5

    with sr.Microphone() as source:
        print("Speak now... you have up to 5 minutes. You can pause for a few seconds.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=20, phrase_time_limit=300)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text

    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""

    except sr.RequestError:
        print("Speech service unavailable")
        return ""

def categorize_email(text):
    text_lower = text.lower()

    spam_keywords = [
        "win money",
        "lottery",
        "free prize",
        "click here",
        "urgent offer"
    ]

    urgent_keywords = [
        "urgent",
        "asap",
        "immediately",
        "important",
        "deadline"
    ]

    work_keywords = [
        "meeting",
        "project",
        "client",
        "report",
        "office"
    ]

    for word in spam_keywords:
        if word in text_lower:
            return "Spam"

    for word in urgent_keywords:
        if word in text_lower:
            return "Urgent"

    for word in work_keywords:
        if word in text_lower:
            return "Work"

    return "General"

def summarize_email(text):
    try:
        prompt = f"Summarize this email:\n{text}\nSummary:"
        inputs = summarizer_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        output_ids = summarizer_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=120,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        generated_text = summarizer_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        return generated_text.strip()

    except Exception as e:
        return f"Summary Error: {e}"

def detect_language(text):
    try:
        return detect(text)

    except:
        return "unknown"

def generate_reply(text):
    try:
        prompt = f"Reply professionally to this email:\n{text}\nReply:"
        inputs = generator_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        output_ids = generator_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=120,
            pad_token_id=generator_tokenizer.eos_token_id,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        generated_text = generator_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        return generated_text.strip()

    except Exception as e:
        return f"Reply Generation Error: {e}"

def send_email(sender_email, sender_password, receiver_email, subject, body):
    msg = MIMEText(body)

    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(
                sender_email,
                receiver_email,
                msg.as_string()
            )
    except smtplib.SMTPAuthenticationError as auth_err:
        raise RuntimeError(
            "SMTP authentication failed. Verify your Gmail address and App Password, "
            "and make sure 2-Step Verification is enabled for the account."
        ) from auth_err
    except smtplib.SMTPException as smtp_err:
        raise RuntimeError(
            f"Failed to send email: {smtp_err}"
        ) from smtp_err

def main():
    print("\nVOICE CONTROLLED EMAIL AI ASSISTANT\n")

    sender_email = input("Enter sender Gmail: ")

    sender_password = input(
        "Enter Gmail App Password: "
    )

    receiver_email = input("Enter receiver email: ")

    subject = input("Enter subject: ")

    print("\nChoose Input Mode")
    print("1. Voice Input")
    print("2. Text Input")

    choice = input("Enter choice: ")

    if choice == "1":
        email_content = listen()

    else:
        email_content = input(
            "Enter email content: "
        )

    if not email_content:
        print("No email content detected")
        return

    category = categorize_email(email_content)

    summary = summarize_email(email_content)

    language = detect_language(email_content)

    reply = generate_reply(email_content)

    print("\nEMAIL ANALYSIS")
    print("Category:", category)
    print("Language:", language)
    print("Summary:", summary)

    print("\nSUGGESTED REPLY\n")
    print(reply)

    speak("Email analysis completed")

    send_choice = input(
        "\nDo you want to send this email? (yes/no): "
    )

    if send_choice.lower() == "yes":
        try:
            send_email(
                sender_email,
                sender_password,
                receiver_email,
                subject,
                email_content
            )
            print("Email sent successfully")
            speak("Email sent successfully")
        except RuntimeError as err:
            print(f"Could not send email: {err}")
            speak("Failed to send email")
    else:
        print("Email not sent")

if __name__ == "__main__":
    main()