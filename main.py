import streamlit as st
import os
import numpy as np
import pandas as pd
import cv2
import math
import numpy as np
import random
from utils import *
import matplotlib.pyplot as plt
from PIL import Image
from grader import *
from tkinter import Tk   
from tkinter.filedialog import askopenfilename, askdirectory
from tqdm import tqdm
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()
EMAIL = os.getenv('EMAIL')
PASSW = os.getenv('PASSW')

# print("Choose folder:")
# path = askdirectory()
path = 'D:/code/.contest/dgnl/thithulan1'
files = os.listdir(path)
dapan = f"{path}/_dapan.jpg"

dapan_infors = get_informations(dapan)
bailam = []
for file in files:
    if ('_dapan' not in file):
        bailam.append(get_informations(f'{path}/{file}'))

dapan_vector = dapan_infors[2]
bailam_matrix = []
for _ in bailam:
    bailam_matrix.append(_[2])
dapan_vector = np.array([dapan_vector])
bailam_matrix = np.array(bailam_matrix)

# JUST FOR TESTING

file = open("answers.csv", "r", encoding='utf8')
data = file.read()
file.close()
bailam = []
bailam_matrix = []
for i in data.split('\n')[2:]:
    hehe = tuple(i[i.find('.com,')+5:i.find(' /')].split(','))
    content = i[i.find('1200,')+5:i.find(',,')].split(',')
    bailam_matrix.append(i[i.find('1200,')+5:i.find(',,')].split(','))
    bailam.append((hehe[0], hehe[1], content))
bailam_matrix = np.array(bailam_matrix)
dapan_vector = np.array([bailam_matrix[0, :]])
bailam = bailam[1:]
bailam_matrix = bailam_matrix[1:, :]

# END JUST FOR TESTING

def cal_p(dapan_vector, bailam_matrix):
    matrix = bailam_matrix.copy()
    n = matrix.shape[0]
    p = []
    for i in range(matrix.shape[0]):
        matrix[i] = matrix[i] == dapan_vector
    for i in range(dapan_vector.shape[1]):
        p.append(sum(matrix[:, i] == 'T') / n)
    return p
def f(p_):
    return math.log((p_+0.001)/(1-p_ + 0.001))
def irt(p):
    weights = []
    for i in p:
        weights.append(f(i))
    weights = list(map(lambda x : x - (sum(weights)/len(p)), weights))
    mx = np.max(weights)
    return list(map(lambda x : x * (3 / mx) + 10, weights))

p_ = cal_p(dapan_vector, bailam_matrix)
van_p = p_[:40]
toan_p = p_[40:70]
kh_p = p_[70:]
van_irt = irt(van_p)
toan_irt = irt(toan_p)
kh_irt = irt(kh_p)
irt_ = van_irt + toan_irt + kh_irt

# final_point = (bailam_matrix == dapan_vector)@np.array(irt_)
van_points = (bailam_matrix[:, :40] == dapan_vector[:, :40])@np.array(van_irt)
toan_points = (bailam_matrix[:, 40:70] == dapan_vector[:, 40:70])@np.array(toan_irt)
kh_points = (bailam_matrix[:, 70:] == dapan_vector[:, 70:])@np.array(kh_irt)

socaudung = sum((bailam_matrix == dapan_vector).T)

for i in range(len(van_points)):
    bailam[i] += (np.round(van_points[i], 2), np.round(toan_points[i], 2), np.round(kh_points[i], 2))

st.title("Thi thử đánh giá năng lực")

df = {"ID" : [], "Question_ID" : [], "Correct" : [], "Languages" : [], "Mathematics" : [], "Problems solving" : [], "Total" : []}
for _, i in enumerate(bailam):
    df["ID"].append(i[0])
    df["Question_ID"].append(i[1])
    df["Correct"].append(socaudung[_])
    df["Languages"].append(i[3])
    df["Mathematics"].append(i[4])
    df["Problems solving"].append(i[5])
    df["Total"].append(i[3] + i[4] + i[5])

st.table(pd.DataFrame(df))
IDs = df["ID"]
view = st.selectbox("chọn bài cần xem:", tuple(IDs))

def get_answer_sheet(bailam, IDs, id, figname = None):
    student = bailam[IDs.index(id)]
    neural = {"x" : [], "y" : []}
    right = {"x" : [], "y" : []}
    wrong = {"x" : [], "y" : []}
    begin_x = 0
    DAPAN = ['A', 'B', 'C', 'D']
    def trung(dapan, x, y, delta = 1):
        for i, d in enumerate(DAPAN):
            if (d == dapan):
                right["x"].append(x + i*delta)
                right["y"].append(y)
            else:
                neural["x"].append(x + i*delta)
                neural["y"].append(y)
            
    def truot(bailam, dapan, x, y, delta = 1):
        for i, d in enumerate(DAPAN):
            if (d == bailam):
                wrong["x"].append(x + i*delta)
                wrong["y"].append(y)
            elif (d == dapan):
                right["x"].append(x + i*delta)
                right["y"].append(y)
            else:
                neural["x"].append(x + i*delta)
                neural["y"].append(y)

    def sobaodanh(sbd):
        begin_x = 29
        begin_y = 42
        for x in range(6):
            for y in range(10):
                neural["x"].append(begin_x - x)
                neural["y"].append(begin_y - y)
        for x in range(6):
            k = int(sbd[x])
            right["x"].append(begin_x - (6-x) + 1)
            right["y"].append(begin_y - k)

    def made(made):
        begin_x = 33
        begin_y = 42
        for x in range(3):
            for y in range(10):
                neural["x"].append(begin_x - x)
                neural["y"].append(begin_y - y)
        for x in range(3):
            k = int(made[x])
            right["x"].append(begin_x - (3-x) + 1)
            right["y"].append(begin_y - k)

    fig, ax = plt.subplots(1, 1, figsize = (5, 7))
    begin_y = 30
    num = 1
    for bailam, dapan in zip(student[2], dapan_vector[0, :]):
        if (bailam == dapan):
            trung(bailam, begin_x, begin_y)
            plt.text(begin_x-1, begin_y-0.3, str(num), horizontalalignment = 'right', fontfamily = 'monospace')
            num += 1
        else:
            truot(bailam, dapan, begin_x, begin_y)
            plt.text(begin_x-1, begin_y-0.3, str(num), horizontalalignment = 'right', color = 'red', fontfamily = 'monospace')
            num += 1
        begin_y -= 1
        if (begin_y == 0):
            begin_x += 10
            begin_y = 30
    sobaodanh(student[0])
    made(student[1])

    x1 = -2
    x2 = 23
    y1 = 42.15
    y2 = 32.8
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color = 'black')
    ax.scatter(neural["x"], neural["y"], color = "black", alpha=0.1)
    ax.scatter(right["x"], right["y"], color = "blue")
    ax.scatter(wrong["x"], wrong["y"], color = "red")
    ax.axis("off")
    plt.xlim([-4, 35])
    plt.ylim([0, 44])
    if (figname != None):
        plt.savefig(f'{student[0]}.png', dpi=600, bbox_inches='tight')
    return fig
    
st.pyplot(get_answer_sheet(bailam, IDs, view))

def send_email(subject, body, sender, recipients, password, id):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with open(f'{id}.png', 'rb') as f:
        img_data = f.read()
    image = MIMEImage(img_data, name=os.path.basename(f'{id}.png'))
    msg.attach(MIMEText(body))
    msg.attach(image)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
       smtp_server.login(sender, password)
       smtp_server.sendmail(sender, recipients, msg.as_string())
    print(f"Message sent to {recipients[0]}!")

if st.button("Send all results!"):
    subject = "[noreply] Kết quả thi thử đánh giá năng lực!"
    sender = EMAIL
    password = PASSW
    print(sender, password)
    file = open("contestants.json")
    contestants = json.load(file)
    file.close()
    for i in tqdm(range(len(list(IDs))), desc = 'Sending email ...'):
        student = bailam[i]
        body = f"Kết quả làm bài của thí sinh {student[0]}|{student[1]} \nSố câu đúng: {socaudung[i]} \nĐiểm phần ngôn ngữ: {student[3]} \nĐiểm phần toán: {student[4]} \nĐiểm phần xử lý vấn đề: {student[5]} \nTổng điểm: {student[3] + student[4] + student[5]}"
        recipients = []
        try:
            recipients.append(contestants[student[0]])
            print(body, recipients)
        except:
            print(f"Cannot find email of contestant {student[0]}!")
            continue

        fig = get_answer_sheet(bailam, IDs, student[0], 1)
        send_email(subject, body, sender, recipients, password, student[0])
